from typing import Sequence, List, TypedDict, Tuple, Optional

import torch

from .model import Transformer
from .generation import Llama, sample_top_p
from .tokenizer import ChatFormat, Message, Tokenizer, Role

class Node(TypedDict):
    requirements: List[int]

class Task(Node):
    expects: Tuple[Role, str]

class Prompt(Node):
    message: Message

class Workflow:

    @staticmethod
    def build(*args, **kwargs) -> "Workflow":
        llama = Llama.build(*args, **kwargs)
        return Workflow(llama.model, llama.tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        self.cur_id = 0
        self.device = "cuda"
        self.id_map = torch.tensor([-1], dtype=torch.long, device=self.device)
        self.position_map = torch.tensor([0], dtype=torch.long, device=self.device)
        self.context = torch.tensor([self.tokenizer.bos_id], dtype=torch.long, device=self.device)

    def reset(self, *args, **kwargs):
        self.cur_id = 0
        self.id_map = torch.tensor([-1], dtype=torch.long, device=self.device)
        self.position_map = torch.tensor([0], dtype=torch.long, device=self.device)
        self.context = torch.tensor([self.tokenizer.bos_id], dtype=torch.long, device=self.device)

    def insert(self, prompts: Sequence[Prompt]) -> List[int]:
        prompt_tokens = []
        prompt_length = []
        for prompt in prompts:
            message = self.formatter.encode_message(prompt['message'])
            prompt_tokens.extend(message)
            prompt_length.append(len(message))

        prompt_tokens = torch.tensor(prompt_tokens, device=self.device)
        prompt_length = torch.tensor(prompt_length, device=self.device)

        prompt_mask = self._dependency_mask(prompts)
        new_ids = torch.repeat_interleave(
            torch.arange(self.cur_id, self.cur_id + len(prompts), device=self.device),
            prompt_length
        )
        full_mask = torch.hstack([
            grouped_causal_mask(new_ids),
            torch.repeat_interleave(prompt_mask, prompt_length, dim=0)
        ])
        position_ids = incremental_sequence_with_offset(
            torch.sum(prompt_mask == 0, dim=1),
            prompt_length
        )

        if self.cur_id == 0:
            self.model.forward(
                tokens=torch.cat([self.context, prompt_tokens]).unsqueeze(0),
                start_pos=0,
                mask=torch.vstack([torch.zeros((1, 1 + len(prompt_tokens)), device=self.device), full_mask]),
                position_ids=torch.hstack([torch.tensor(0, device=self.device), position_ids])
            )
        else:
            self.model.forward(
                tokens=prompt_tokens.unsqueeze(0),
                start_pos=len(self.context),
                mask=full_mask,
                position_ids=position_ids
            )

        self.id_map = torch.cat([self.id_map, new_ids])
        self.position_map = torch.cat([self.position_map, position_ids])
        self.context = torch.cat([self.context, prompt_tokens])
        self.cur_id += len(prompts)

        return list(range(self.cur_id - len(prompts), self.cur_id))

    def _dependency_mask(self, nodes: Sequence[Node]) -> torch.Tensor:
        mask = torch.full((len(nodes), len(self.context)), float("-inf"), device=self.device)
        for i, node in enumerate(nodes):
            meets_requirement = torch.isin(
                self.id_map,
                torch.tensor(node['requirements'], dtype=torch.long, device=self.device)
            )
            is_identity = (self.id_map == (self.cur_id + i))
            mask[i, meets_requirement | is_identity] = 0
        mask[:, 0] = 0 # bos
        return mask

    @torch.inference_mode()
    def step(
        self,
        tasks: List[Task],
        compact: bool = False,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        log_probs: bool = True,
        prefill: bool = True,
        seed: int = 42
    ) -> Tuple[List[List[int]], List[int], Optional[List[List[float]]]]:
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        # TODO -- runtime validations for remaining output buffer size
        bsz = len(tasks)
        pad_id = self.tokenizer.pad_id

        # TODO -- this doesn't need to be a separate buffer
        tokens = torch.full((1, bsz * max_gen_len), pad_id, device=self.device)

        """
        TODO -- if we compact and produce a multi-task alignment, then the
        postion ids might not left-compacted.

        For example, say the context is A B C and we want to generate both
        A B -> D as well as B C -> E in parallel. Rather than position both
        A B and B C at position 0, thus losing parallelism, we can just line
        them up as A B C sequentially. Then, we can generate D from offset
        len(A) + len(B) and E from offset len(A) + len(B) + len(C):

          pos 0
            v
            <message A><message B><message C>
                                    <message D><message E> <- in parallel!

        TODO -- there's a few too many indirections and allocations here for
        this to be really clean and fast, but the way it is now is flexible.
        """
        mask = self._dependency_mask(tasks)
        if compact:
            if len(tasks) > 1:
                raise NotImplementedError()
            where = torch.where(mask[0] == 0)[0]
            from_pos = self.position_map[where]
            from_ids = self.id_map[where]
            to_pos = torch.empty(len(from_pos), dtype=torch.long, device=self.device)
            to_pos[0] = 0 # bos
            offset = 1
            for req in tasks[0]['requirements']:
                req_length = torch.sum(from_ids == req)
                req_position_ids = torch.where(from_ids == req)[0]
                to_pos[req_position_ids] = torch.arange(offset, offset + req_length)
                offset += req_length
            self.model.reposition_cache(where, from_pos, to_pos)
            self.position_map[where] = to_pos
        position_ids = torch.sum(mask == 0, dim=1) # (bsz,)

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
            for task in tasks:
                role = f"{task['expects'][0]}:{task['expects'][1]}" if task['expects'][1] else task['expects'][0]
                header = self.formatter.encode_header({"role": role}) # type: ignore
                prefill_tokens.extend(header)
                prefill_length.append(len(header))

            prefill_tokens = torch.tensor(prefill_tokens, device=self.device)
            prefill_length = torch.tensor(prefill_length, device=self.device)

            new_ids = torch.repeat_interleave(
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device),
                prefill_length
            )

            grouped = grouped_causal_mask(new_ids)
            requirements = torch.repeat_interleave(mask, prefill_length, dim=0)
            full_mask = torch.hstack([requirements, grouped])
            prefill_position_ids = incremental_sequence_with_offset(position_ids, prefill_length)

            prefill_logits = self.model.forward(
                tokens=prefill_tokens.unsqueeze(0),
                start_pos=len(self.context),
                mask=full_mask,
                position_ids=prefill_position_ids
            )

            self.id_map = torch.cat([self.id_map, new_ids])
            self.position_map = torch.cat([self.position_map, prefill_position_ids])
            self.context = torch.cat([self.context, prefill_tokens])

            # update for decoding loop
            mask = self._dependency_mask(tasks)
            position_ids += prefill_length

        eos_reached = torch.tensor([False] * bsz, device=self.device)
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)

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

            if cur_pos == 0 and prefill:
                logits = prefill_logits[:, torch.cumsum(prefill_length, dim=0) - 1]
            else:
                logits = self.model.forward(
                    tokens=tokens[:, cur_pos - bsz : cur_pos],
                    start_pos=len(self.context) - bsz,
                    mask=mask,
                    position_ids=position_ids
                )

            if temperature > 0:
                probs = torch.softmax(logits[:, -bsz:] / temperature, dim=-1)
                next_token = sample_top_p_parallel(probs, top_p, generator=generator).squeeze(0)
            else:
                next_token = torch.argmax(logits[:, -bsz:], dim=-1).squeeze(0)

            tokens[:, cur_pos : cur_pos + bsz][:, ~eos_reached] = next_token[~eos_reached]

            self.id_map = torch.cat([self.id_map, torch.where(
                eos_reached,
                torch.full((bsz,), pad_id, device=self.device),
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)
            )])
            self.position_map = torch.cat([self.position_map, position_ids])
            self.context = torch.cat([self.context, tokens[:, cur_pos : cur_pos + bsz].squeeze(0)])

            eos_reached |= torch.isin(next_token, stop_tokens)
            interleaved_mask = torch.full((bsz, bsz), float("-inf")).type_as(mask)
            interleaved_mask.fill_diagonal_(0)
            mask = torch.hstack([mask, interleaved_mask])
            position_ids += 1

            if all(eos_reached):
                break

        tokens = tokens.view(-1, bsz).t()
        out_tokens = []
        out_ids = []
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

def sample_top_p_parallel(probs, p, generator=None):
    next_token = sample_top_p(probs.view(-1, probs.shape[-1]), p, generator)
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
