import warnings
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
    messages: List[Message]

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
        self.device = device
        self.stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)
        self.max_seq_len = self.model.params.max_seq_len
        self.max_nodes = max_nodes
        self.max_parents = max_parents
        self.model.forward(torch.tensor([self.tokenizer.bos_id], device=self.device).unsqueeze(0), 0) # set the cache for bos just oncez
        self.reset()

    def reset(
        self,
        new_max_seq_len: Optional[int] = None,
        new_max_nodes: Optional[int] = None,
        new_max_parents: Optional[int] = None
    ):
        if new_max_nodes is not None:
            self.max_nodes = new_max_nodes
        if new_max_parents is not None:
            self.max_parents = new_max_parents
        if new_max_seq_len is not None:
            self.max_seq_len = new_max_seq_len
        self.cur_id = 1
        self.cache_len = 1
        self.node_map = torch.full((self.max_seq_len,), -1, dtype=torch.long, device=self.device)
        self.node_map[0] = 0 # bos
        self.position_map = torch.zeros((self.max_seq_len,), dtype=torch.long, device=self.device)
        self.context = torch.full((self.max_seq_len,), self.tokenizer.bos_id, dtype=torch.long, device=self.device)
        self.parent_map = torch.zeros(
            self.max_nodes, self.max_parents, dtype=torch.long, device=self.device
        ) # implicitly bos is always a parent
        self.parent_map[:, 0] = torch.arange(
            self.max_nodes, dtype=torch.long, device=self.device
        ) # every node is its own parent too

    # TODO -- we should make this lazy
    def insert(self, prompts: Sequence[Prompt]) -> List[Cached]:
        self.add_nodes(prompts)
        prompt_tokens = []
        prompt_length = []
        outputs = []
        for i, prompt in enumerate(prompts):
            tokens = self.formatter.encode_dialog(prompt['messages'])
            outputs.append({
                'id': self.cur_id + i,
                'parent_ids': prompt['parent_ids'],
                'tokens': tokens,
                'length': len(tokens)
            })
            prompt_length.append(len(tokens))
            prompt_tokens.extend(tokens)
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
        seed: int = 42,
        debug: bool = False,
    ) -> Tuple[List[List[int]], List[Cached]]:
        bsz = len(tasks)
        if self.cache_len + (bsz * max_gen_len) > self.max_seq_len:
            raise Exception(f"Insufficient capacity for {bsz * max_gen_len} tokens.")
        if self.cur_id + bsz > self.max_nodes:
            raise Exception(f"Insufficient capacity for {bsz * max_gen_len} nodes.")

        self.add_nodes(tasks)
        mask = self.dynamic_mask(self.parent_map[self.cur_id : self.cur_id + bsz])
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        if compact:
            if len(tasks) > 0:
                warnings.warn("Multi-node compaction not fully implemented. Use caution.")
            self.compact(tasks[0]['parent_ids'], mask[0]) # use just the first task for now

        # tokenize headers
        headers = []
        header_length = []
        for i, task in enumerate(tasks):
            role = f"{task['expects'][0]}:{task['expects'][1]}" if task['expects'][1] else task['expects'][0]
            header = self.formatter.encode_header({"role": role})
            headers.append(header)
            header_length.append(len(header))

        # recompute mask and position_ids after prefilling
        prefill_logits = self.prefill(sum(headers, []), header_length, cached_mask=mask)
        mask = self.dynamic_mask(self.parent_map[self.cur_id : self.cur_id + bsz])
        position_ids = torch.sum(mask == 0, dim=1)

        if debug:
            print('Before decoding...')
            self.debug_mask(mask)

        start_pos = self.cache_len
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        mask = self.preallocate_interleaved_mask(mask, bsz, max_gen_len)

        for cur_pos in range(0, bsz * max_gen_len - bsz, bsz): # do N - 1 iterations
            if cur_pos == 0:
                header_ends = torch.cumsum(torch.tensor(header_length, device=self.device), dim=0) - 1
                logits = prefill_logits[:, header_ends]
            else:
                logits = self.model.forward(
                    tokens=self.context[self.cache_len - bsz : self.cache_len].unsqueeze(0),
                    start_pos=self.cache_len - bsz,
                    mask=mask[:, : self.cache_len],
                    position_ids=position_ids
                )

            if temperature > 0:
                probs = torch.softmax(logits[:, -bsz:] / temperature, dim=-1)
                next_token = sample_top_p_parallel(probs, top_p, generator=generator).squeeze(0)
            else:
                next_token = torch.argmax(logits[:, -bsz:], dim=-1).squeeze(0)

            self.node_map[self.cache_len : self.cache_len + bsz][~eos_reached] = \
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)[~eos_reached]
            self.context[self.cache_len : self.cache_len + bsz][~eos_reached] = next_token[~eos_reached]
            self.position_map[self.cache_len : self.cache_len + bsz] = position_ids

            position_ids += 1
            self.cache_len += bsz
            eos_reached |= torch.isin(next_token, self.stop_tokens)
            if all(eos_reached):
                break

        # TODO -- force decode eot_id for everything that didn't naturally terminate
        # self.node_map[self.cache_len : self.cache_len + bsz][~eos_reached] = \
        #     torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)[~eos_reached]
        # self.context[self.cache_len : self.cache_len + bsz][~eos_reached] = 128009 # eot_id
        # self.position_map[self.cache_len : self.cache_len + bsz] = position_ids
        # self.cache_len += bsz

        if debug:
            print('After decoding...')
            self.debug_mask(mask[:, : self.cache_len])

        # one more forward pass to top off the kv cache
        self.model.forward(
            tokens=self.context[self.cache_len - bsz : self.cache_len].unsqueeze(0),
            start_pos=self.cache_len - bsz,
            mask=mask[:, : self.cache_len],
            position_ids=position_ids
        )

        self.cache_len = start_pos if stateless else self.cache_len
        outputs = self.wrap_outputs(
            self.context[start_pos : self.cache_len].view(-1, bsz).t(),
            tasks,
            headers
        ) # order matters here (which isn't great design)
        self.cur_id += 0 if stateless else bsz
        return outputs

    def wrap_outputs(
        self,
        tokens: torch.Tensor,
        tasks: List[Task],
        headers: List[List[int]]
    ) -> Tuple[List[List[int]], List[Cached]]:
        out_tokens = []
        out_nodes = []
        for i, (task, toks, header) in enumerate(zip(tasks, tokens.tolist(), headers)):
            found_stop = False
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx+1] # include the stop token
                    out_tokens.append(toks[:-1])
                    out_nodes.append({
                        'id': self.cur_id + i,
                        'parent_ids': task['parent_ids'],
                        'tokens': header + toks,
                        'length': len(header) + len(toks)
                    })
                    found_stop = True
                except ValueError:
                    pass
            if not found_stop:
                out_tokens.append(toks)
                out_nodes.append({
                    'id': self.cur_id + i,
                    'parent_ids': task['parent_ids'],
                    'tokens': header + toks,
                    'length': len(header) + len(toks)
                })
        return out_tokens, out_nodes

    def add_nodes(self, nodes: Sequence[Node]) -> torch.Tensor:
        for i, node in enumerate(nodes):
            self.parent_map[self.cur_id + i, 1 : 1 + len(node['parent_ids'])] = \
                torch.tensor(node['parent_ids'], device=self.device)
        return self.parent_map[self.cur_id : self.cur_id + len(nodes)]

    def prefill(
        self,
        _tokens: List[int],
        _length: List[int],
        cached_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = len(_length)
        tokens = torch.tensor(_tokens, device=self.device)
        length = torch.tensor(_length, device=self.device)

        # important invariant: self.cur_id is only mutated AFTER prefilling
        node_par_mask = cached_mask if cached_mask is not None else \
            self.dynamic_mask(self.parent_map[self.cur_id : self.cur_id + B])
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

    def compact(self, order: List[int], mask: torch.Tensor):
        # (num_parent_tokens,)
        where = torch.where(mask.squeeze() == 0)[0]
        from_pos = self.position_map[where]
        from_nodes = self.node_map[where]
        to_pos = torch.zeros(len(from_pos), dtype=torch.long, device=self.device)

        offset = 1
        for par in order:
            par_mask = (from_nodes == par)
            par_length = torch.sum(par_mask)
            to_pos[par_mask] = torch.arange(offset, offset + par_length) # type: ignore
            offset += par_length

        self.model.reposition_cache(where, from_pos, to_pos)
        self.position_map[where] = to_pos

    def preallocate_interleaved_mask(self, base_mask: torch.Tensor, bsz: int, max_gen_len: int):
        interleaved_mask = torch.full((bsz, bsz), float("-inf"))
        interleaved_mask.fill_diagonal_(0)
        return torch.hstack([base_mask, interleaved_mask.repeat(1, max_gen_len)])

    def dynamic_mask(self, node_parents: torch.Tensor) -> torch.Tensor:
        B, _ = node_parents.shape
        return torch.where(
            (self.node_map[:self.cache_len].unsqueeze(0).unsqueeze(2) == node_parents.unsqueeze(1)).any(dim=2),
            torch.zeros(B, self.cache_len, device=self.device),
            torch.full((B, self.cache_len), float("-inf"), device=self.device)
        )

    def debug_mask(self, mask: torch.Tensor):
        for i, thread in enumerate((mask == 0)):
            print(f'Output thread {i}:')
            print('#' * 20)
            print(self.tokenizer.decode(self.context[:self.cache_len][thread].tolist()))
            print('#' * 20)

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
