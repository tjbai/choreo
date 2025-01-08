# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer, Role


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


class Task(TypedDict):
    requirements: List[int]
    expects: Tuple[Role, str]

class Workflow(Llama):

    # TODO -- shouldn't be hardcoded
    BOS_ID = 128000
    BOT_ID = 128006
    EOT_ID = 128009

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_id = 0
        self.id_map = torch.tensor([-1], dtype=torch.long, device="cuda")
        self.context = torch.tensor([self.BOS_ID], dtype=torch.long, device="cuda")
        self.device = "cuda"

    def insert(self, dialog: Dialog) -> List[int]:
        prev_id = self.cur_id
        self._insert(self.formatter.encode_dialog_prompt(dialog, prefill=False))
        assert self.cur_id - prev_id == len(dialog)
        return list(range(self.cur_id - len(dialog), self.cur_id))

    # TODO -- implement masking analogous to step
    def _insert(self, token_ids: List[int]):
        if token_ids[0] == self.BOS_ID:
            token_ids = token_ids[1:]
        tokens = torch.tensor(token_ids, dtype=torch.long)

        eot_increment = torch.cumsum(tokens == self.BOT_ID, dim=0)
        new_ids = torch.full_like(tokens, self.cur_id) + eot_increment - 1
        self.cur_id += eot_increment[-1].item()

        self.model.forward(tokens, len(self.context))
        self.id_map = torch.cat([self.id_map, new_ids])
        self.context = torch.cat([self.context, tokens])

    def _dependency_mask(self, tasks):
        mask = torch.full((len(tasks), len(self.context)), float("-inf"), device=self.device)
        for i, task in enumerate(tasks):
            meets_requirement = torch.isin(self.id_map, torch.tensor(task['requirements'], device=self.device))
            is_identity = (self.id_map == (self.cur_id + i))
            mask[i, meets_requirement | is_identity] = 0
        mask[:, 0] = 0 # bos
        return mask

    @torch.inference_mode()
    def step(
        self,
        tasks: List[Task],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        log_probs: bool = True,
        prefill: bool = True
    ) -> Tuple[List[List[int]], List[int], Optional[List[List[float]]]]:

        # TODO -- runtime validations
        bsz = len(tasks)
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((1, bsz * max_gen_len), pad_id, dtype=torch.long, device="cuda")
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device="cuda")

        # TODO -- this only left-compacts each new generation without repositioning the context
        mask = self._dependency_mask(tasks)
        position_ids = torch.cumsum(mask == 0, dim=1) # (bsz,)

        """
        We might want to prefill the start of each message with the expected role
        and special tokens. Rather than interleave, we can concatenate them to the
        context sequentially:

            <context><prefill 1><prefill 2>...

        To add them to the cache, we just need to do one forward pass. The tricky
        part is correctly handling the attention mask because each prefill needs to:
            1. only have causal attention over other tokens in the prefill sequence
            2. and only attend over required tokens in the context.
        """
        if prefill:
            prefill_tokens = []
            prefill_length = []
            for i, task in enumerate(tasks):
                role = f"{task['expects'][0]}:{task['expects'][1]}" if task['expects'][1] else task['expects'][0]
                header = self.formatter.encode_header({"role": role}) # type: ignore
                prefill_tokens.extend(header)
                prefill_length.append(len(header))

            prefill_tokens = torch.tensor(prefill_tokens, device=self.device)
            prefill_length = torch.tensor(prefill_length, device=self.device)

            # (seqlen, seqlen)
            grouped = grouped_causal_mask(torch.tensor(prefill_tokens))

            # (seqlen, cachelen)
            requirements = torch.repeat_interleave(mask, prefill_length, dim=0)

            self.model.forward(
                tokens=prefill_tokens.unsqueeze(0),
                start_pos=len(self.context),
                mask=torch.hstack([requirements, grouped]),
                position_ids=incremental_sequence_with_offset(position_ids, prefill_length)
            )

            self.id_map = torch.cat([
                self.id_map,
                torch.repeat_interleave(
                    torch.arange(self.cur_id, self.cur_id + bsz),
                    prefill_length
                )
            ])
            self.context = torch.cat([self.context, prefill_tokens])

            # need to update these with the new tokens
            mask = self._dependency_mask(tasks)
            position_ids += prefill_length

        # TODO -- this is gross and probably not necessary
        tail_mask = -((mask == 0).flip(-1).int().argmax(-1) - len(self.context))

        for cur_pos in range(0, bsz * max_gen_len, bsz):
            """
            In each step, (cache_len + i) should only be able to attend over
            (cache_len - bsz + i) from the previous set of interleaved generations.

            For example, at batch size 3, this looks like:

                 previous  current
                     v       v
                   1 0 0   1 0 0
            mask + 0 1 0 + 0 1 0
                   0 0 1   0 0 1
            """

            tokens = (
                self.context[torch.arange(bsz, device=self.device), tail_mask]
                if cur_pos == 0 else tokens[:, cur_pos : cur_pos + bsz]
            )

            logits = self.model.forward(
                tokens=tokens,
                start_pos=len(self.context) + cur_pos,
                mask=mask,
                position_ids=position_ids
            )

            if temperature > 0:
                probs = torch.softmax(logits[:, -bsz:] / temperature, dim=-1)
                next_token = sample_top_p_parallel(probs, top_p).squeeze()
            else:
                next_token = torch.argmax(logits[:, -bsz:], dim=-1)

            eos_reached |= torch.isin(next_token, stop_tokens)
            tokens[:, cur_pos : cur_pos + bsz][eos_reached] = next_token[eos_reached]

            interleaved_mask = torch.full((bsz, bsz), float("-inf")).type_as(mask)
            interleaved_mask.fill_diagonal_(0)
            mask = torch.hstack([mask, interleaved_mask])

            new_ids = torch.where(
                eos_reached,
                torch.full((bsz,), pad_id, device=self.device),
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)
            )
            self.id_map = torch.cat([self.id_map, new_ids])

            position_ids += 1

            if all(eos_reached):
                break

        tokens = tokens.view(-1, bsz).t()
        out_tokens, out_ids = []
        for i, toks in enumerate(tokens.tolist()):
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_ids.append(self.cur_id)
            self.cur_id += 1

        return out_tokens, out_ids, None


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_p_parallel(probs, p):
    """
    Akin to `sample_top_p` but expecting a BxNxV shape tensor for probs.
    Returns a shape BxN tensor of samples.
    """
    next_token = sample_top_p(probs.view(-1, probs.shape[-1]), p)
    return next_token.view(probs.shape[0], probs.shape[1])


def grouped_causal_mask(message_ids: torch.Tensor) -> torch.Tensor:
    seqlen = len(message_ids)
    causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device=message_ids.device)
    causal = torch.tril(causal)
    same_message = message_ids.unsqueeze(0) == message_ids.unsqueeze(1)
    return torch.where(causal & same_message, torch.zeros(seqlen, seqlen), torch.full((seqlen, seqlen), float("-inf")))


def incremental_sequence_with_offset(offsets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    assert len(offsets) == len(lengths)
    position_ids = torch.repeat_interleave(offsets, lengths)
    start = 0
    for n in lengths.tolist():
        position_ids[start : start + n] += torch.arange(n, device=position_ids.device)
        start += n
    return position_ids
