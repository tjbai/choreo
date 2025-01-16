from typing import Sequence, List, TypedDict, Tuple, Optional

import torch

from .model import Transformer
from .generation import Llama, sample_top_p
from .tokenizer import ChatFormat, Message, Tokenizer, Role

class Node(TypedDict):
    parent_ids: List[int]

class Task(Node):
    expects: Tuple[Role, Optional[str]]

class Prompt(Node):
    message: Message

class Cached(Node):
    id: int
    tokens: List[int]
    length: int

class Workflow:
    @staticmethod
    def build(*args, max_nodes: int = 25, max_parents: int = 25, **kwargs) -> "Workflow":
        llama = Llama.build(*args, **kwargs)
        return Workflow(llama.model, llama.tokenizer, max_nodes, max_parents)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        max_nodes: int,
        max_parents: int,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        self.stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)
        self.max_seq_len = self.model.params.max_seq_len
        self.max_nodes = max_nodes
        self.max_parents = max_parents
        self.device = device
        self.model.forward(torch.tensor([self.tokenizer.bos_id], device=self.device).unsqueeze(0), 0) # set the cache for bos just once
        self.reset()

    def reset(self, *args, **kwargs):
        self.cur_id = 1
        self.cache_len = 1
        self.node_map = torch.full((self.max_seq_len,), -1, dtype=torch.long, device=self.device)
        self.node_map[0] = 0 # bos
        self.position_map = torch.zeros((self.max_seq_len,), dtype=torch.long, device=self.device)
        self.context = torch.full((self.max_seq_len,), self.tokenizer.bos_id, dtype=torch.long, device=self.device)
        self.parent_map = torch.zeros(self.max_nodes, self.max_parents, dtype=torch.long, device=self.device) # implicitly bos is always a parent
        self.parent_map[:, 0] = torch.arange(self.max_parents, dtype=torch.long, device=self.device) # every node is its own parent as well

    # TODO -- we should make this lazy
    def insert(self, prompts: Sequence[Prompt]) -> List[Cached]:
        self.register_nodes(prompts)
        prompt_tokens = []
        prompt_length = []
        outputs = []
        for i, prompt in enumerate(prompts):
            message = self.formatter.encode_message(prompt['message'])
            outputs.append({
                'id': self.cur_id + i,
                'parent_ids': prompt['parent_ids'],
                'tokens': message,
                'length': len(message)
            })
            prompt_length.append(len(message))
            prompt_tokens.extend(message)
        self.prefill(prompt_tokens, prompt_length)
        self.cur_id += len(prompts)
        return outputs

    @torch.inference_mode()
    def step(
        self,
        tasks: List[Task],
        compact: bool = False,
        stateless: bool = False,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        log_probs: bool = True,
        seed: int = 42
    ) -> Tuple[List[List[int]], List[Cached]]:
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        self.register_nodes(tasks)

        B = len(tasks)
        if self.cache_len + (B * max_gen_len) > self.max_seq_len:
            raise Exception(f"Output buffers do not have capacity for {B * max_gen_len} tokens.")
        eos_reached = torch.tensor([False] * B, device=self.device)

        # """
        # TODO -- if we compact and produce a multi-task alignment, then the
        # postion ids might not left-compacted.
        # TODO -- there's a few too many indirections and allocations here for
        # this to be really clean and fast, but the way it is now is flexible.
        # """
        # if compact:
        #     if len(tasks) > 1:
        #         print('Not fully implemented, compact with caution!')
        #     where = torch.where(mask[0] == 0)[0]
        #     from_pos = self.position_map[where]
        #     from_ids = self.node_map[where]
        #     to_pos = torch.empty(len(from_pos), dtype=torch.long, device=self.device)
        #     to_pos[0] = 0 # bos
        #     offset = 1
        #     for par in tasks[0]['parent_ids']:
        #         par_length = torch.sum(from_ids == par)
        #         par_position_ids = torch.where(from_ids == par)[0]
        #         to_pos[par_position_ids] = torch.arange(offset, offset + par_length)
        #         offset += par_length
        #     self.model.reposition_cache(where, from_pos, to_pos)
        #     self.position_map[where] = to_pos

        headers = []
        header_length = []
        for i, task in enumerate(tasks):
            role = f"{task['expects'][0]}:{task['expects'][1]}" if task['expects'][1] else task['expects'][0]
            header = self.formatter.encode_header({"role": role}) # type: ignore
            headers.append(header)
            header_length.append(len(header))
        prefill_logits = self.prefill(sum(headers, []), header_length)

        # recompute parent mask and positions after prefilling
        mask = self._parent_mask(self.parent_map[self.cur_id : self.cur_id + B])
        position_ids = torch.sum(mask == 0, dim=1)

        # preallocate mask for decoding
        interleaved_mask = torch.full((B, B), float("-inf"))
        interleaved_mask.fill_diagonal_(0)
        mask = torch.hstack([mask, interleaved_mask.repeat(1, max_gen_len)])

        start_pos = self.cache_len
        for cur_pos in range(0, B * max_gen_len, B):
            if cur_pos == 0:
                logits = prefill_logits[:, torch.cumsum(torch.tensor(header_length, device=self.device), dim=0) - 1]
            else:
                logits = self.model.forward(
                    tokens=self.context[self.cache_len - B : self.cache_len].unsqueeze(0),
                    start_pos=self.cache_len - B,
                    mask=mask[:, : self.cache_len],
                    position_ids=position_ids
                )

            if temperature > 0:
                probs = torch.softmax(logits[:, -B:] / temperature, dim=-1)
                next_token = sample_top_p_parallel(probs, top_p, generator=generator).squeeze(0)
            else:
                next_token = torch.argmax(logits[:, -B:], dim=-1).squeeze(0)

            self.node_map[self.cache_len : self.cache_len + B][~eos_reached] = \
                torch.arange(self.cur_id, self.cur_id + B, device=self.device)[~eos_reached]
            self.context[self.cache_len : self.cache_len + B][~eos_reached] = next_token[~eos_reached]
            eos_reached |= torch.isin(next_token, self.stop_tokens)

            self.position_map[self.cache_len : self.cache_len + B] = position_ids
            position_ids += 1

            self.cache_len +=B
            if all(eos_reached):
                break

        # one more forward pass to top off the kv cache
        self.model.forward(
            tokens=self.context[self.cache_len - B : self.cache_len].unsqueeze(0),
            start_pos=self.cache_len - B,
            mask=mask[:, : self.cache_len],
            position_ids=position_ids
        )

        out_tokens = []
        out_nodes = []
        tokens = self.context[start_pos : self.cache_len].view(-1, B).t()
        for i, (task, toks, header) in enumerate(zip(tasks, tokens.tolist(), headers)):
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx+1] # include the stop token
                except ValueError:
                    pass
            out_tokens.append(toks[:-1])
            out_nodes.append({
                'id': self.cur_id,
                'parent_ids': task['parent_ids'],
                'tokens': header + toks,
                'length': len(header) + len(toks)
            })

        if stateless:
            self.cache_len = start_pos
        else:
            self.cur_id += B

        return out_tokens, out_nodes

    def prefill(self, _tokens: List[int], _length: List[int]) -> torch.Tensor:
        B = len(_length)
        tokens = torch.tensor(_tokens, device=self.device)
        length = torch.tensor(_length, device=self.device)

        # important invariant: self.cur_id is only mutated AFTER prefilling
        node_par_mask = self._parent_mask(self.parent_map[self.cur_id : self.cur_id + B])
        node_pos_ids = torch.sum(node_par_mask == 0, dim=1)

        node_ids = torch.repeat_interleave(
            torch.arange(self.cur_id, self.cur_id + B, device=self.device),
            length
        )
        attn_mask = torch.hstack([
            torch.repeat_interleave(node_par_mask, length, dim=0),
            grouped_causal_mask(node_ids)
        ])
        position_ids = incremental_sequence_with_offset(node_pos_ids, length)

        logits = self.model.forward(
            tokens=tokens.unsqueeze(0),
            start_pos=self.cache_len,
            mask=attn_mask,
            position_ids=position_ids
        )

        N = len(tokens)
        self.node_map[self.cache_len : self.cache_len + N] = node_ids
        self.position_map[self.cache_len : self.cache_len + N] = position_ids
        self.context[self.cache_len : self.cache_len + N] = tokens
        self.cache_len += N

        return logits

    def register_nodes(self, nodes: Sequence[Node]) -> torch.Tensor:
        for i, node in enumerate(nodes):
            self.parent_map[self.cur_id + i, 1 : 1 + len(node['parent_ids'])] = \
                torch.tensor(node['parent_ids'], device=self.device)
        return self.parent_map[self.cur_id : self.cur_id + len(nodes)]

    def _parent_mask(self, parents: torch.Tensor) -> torch.Tensor:
        B, _ = parents.shape
        return torch.where(
            (self.node_map[:self.cache_len].unsqueeze(0).unsqueeze(2) == parents.unsqueeze(1)).any(dim=2),
            torch.zeros(B, self.cache_len, device=self.device),
            torch.full((B, self.cache_len), float("-inf"), device=self.device)
        )

def sample_top_p_parallel(probs, p, generator=None):
    next_token = sample_top_p(probs.view(-1, probs.shape[-1]), p, generator)
    return next_token.view(probs.shape[0], probs.shape[1])

def grouped_causal_mask(message_ids: torch.Tensor) -> torch.Tensor:
    seqlen = len(message_ids)
    causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device=message_ids.device)
    causal = torch.tril(causal)
    same_message = message_ids.unsqueeze(0) == message_ids.unsqueeze(1)
    return torch.where(causal & same_message, torch.zeros(seqlen, seqlen), torch.full((seqlen, seqlen), float("-inf")))

# this is a bit painful
def incremental_sequence_with_offset(offsets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    assert len(offsets) == len(lengths)
    position_ids = torch.repeat_interleave(offsets, lengths)
    start = 0
    for n in lengths.tolist():
        position_ids[start : start + n] += torch.arange(n, device=position_ids.device)
        start += n
    return position_ids

# TODO -- CPU offloading
class CacheManager:
    def __init__(self):
        pass
