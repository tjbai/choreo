import warnings
from typing_extensions import Required
from contextlib import nullcontext
from typing import Sequence, List, TypedDict, Tuple, Optional

import torch

from llama.model import Transformer
from llama.generation import Llama, sample_top_p
from llama.tokenizer import ChatFormat, Message, Tokenizer, Role

class Node(TypedDict):
    parent_ids: List[int]

class Task(Node, total=False):
    header: Required[Tuple[Role, Optional[str]]]
    prefill: Optional[str]

class Prompt(Node):
    messages: List[Message]

class Cached(Node):
    id: int
    output_tokens: List[int]
    output_length: int

class StepResult(TypedDict):
    tokens: List[List[int]]
    nodes: List[Cached]
    log_probs: Optional[List[torch.Tensor]]

class Workflow:
    @staticmethod
    def build(*args, max_nodes: int = 25, **kwargs) -> "Workflow":
        llama = Llama.build(*args, **kwargs)
        return Workflow(llama.model, llama.tokenizer, max_nodes)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        max_nodes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        self.device = device
        self.stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)
        self.max_nodes = max_nodes
        self.reset()

    def reset(self, new_max_nodes: Optional[int] = None):
        # this preamble is important to provide a clean state for autograd
        if self.model.training:
            self.model.cache_k = torch.zeros_like(self.model.cache_k, requires_grad=False)
            self.model.cache_v = torch.zeros_like(self.model.cache_v, requires_grad=False)
            for i, layer in enumerate(self.model.layers):
                layer.attention.cache_k = self.model.cache_k[i]
                layer.attention.cache_v = self.model.cache_v[i]

        self.model.forward(torch.tensor([self.tokenizer.bos_id], device=self.device).unsqueeze(0), 0)

        # node tracking metadata
        if new_max_nodes is not None:
            self.max_nodes = new_max_nodes
        self.max_seq_len = self.model.params.max_seq_len
        self.cur_id = 1
        self.cache_len = 1
        self.node_map = torch.full((self.max_seq_len,), -1, dtype=torch.long, device=self.device)
        self.node_map[0] = 0 # bos
        self.position_map = torch.zeros((self.max_seq_len,), dtype=torch.long, device=self.device)
        self.context = torch.full((self.max_seq_len,), self.tokenizer.bos_id, dtype=torch.long, device=self.device)
        self.adj = torch.eye(self.max_nodes, dtype=torch.bool)
        self.adj[:, 0] = True # bos is always a parent

    def insert(
        self,
        prompts: List[Prompt],
        track_gradients: bool = False,
        time_buffer: Optional[List[int]] = None,
    ) -> List[Cached]:
        start_event = None
        end_event = None
        if time_buffer is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream=None)

        with (nullcontext() if track_gradients else torch.inference_mode()):
            if self.cur_id + len(prompts) > self.max_nodes:
                raise Exception(f"Insufficient capacity for {len(prompts)} more nodes.")
            self.add_nodes(prompts)
            prompt_tokens = []
            prompt_length = []
            outputs = []
            for i, prompt in enumerate(prompts):
                tokens = self.formatter.encode_dialog(prompt['messages'])
                outputs.append(Cached({
                    'id': self.cur_id + i,
                    'parent_ids': prompt['parent_ids'],
                    'output_tokens': tokens,
                    'output_length': len(tokens)
                }))
                prompt_length.append(len(tokens))
                prompt_tokens.extend(tokens)
            if self.cache_len + (new_tokens := sum(prompt_length)) > self.max_seq_len:
                raise Exception(f"Insufficient capacity for {new_tokens} tokens.")
            self.prefill(prompt_tokens, prompt_length, start=self.cur_id, end=self.cur_id+len(prompts))
            self.cur_id += len(prompts)

            if time_buffer is not None:
                end_event.record(stream=None)
                torch.cuda.synchronize()
                time_buffer.append(start_event.elapsed_time(end_event) / 1000.0)

            return outputs

    def _step(
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
        teacher_force: Optional[torch.Tensor] = None,
        time_buffer: Optional[List[int]] = None,
        ttft_buffer: Optional[List[int]] = None,
        force_tokens: Optional[int] = None,
    ) -> StepResult:
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=None)

        bsz = len(tasks)
        if self.cache_len + (bsz * max_gen_len) > self.max_seq_len:
            raise Exception(f"Insufficient capacity for {bsz * max_gen_len} tokens.")
        if self.cur_id + bsz > self.max_nodes:
            raise Exception(f"Insufficient capacity for {bsz} nodes.")

        self.add_nodes(tasks)
        mask = self.dynamic_mask(self.cur_id, self.cur_id + bsz)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        if compact:
            if len(tasks) > 1:
                warnings.warn("Multi-node compaction not fully implemented. Use caution.")
            self.compact(tasks[0]['parent_ids'], mask[0]) # use the ordering from just the first task, for now

        header_start = self.cache_len
        headers, header_length, content_prefills = self.build_headers(tasks)
        prefill_logits = self.prefill(sum(headers, []), header_length, start=self.cur_id, end=self.cur_id+bsz, cached_mask=mask)
        header_end = self.cache_len

        mask = self.dynamic_mask(self.cur_id, self.cur_id + bsz)
        position_ids = self.leftmost_position_ids(mask == 0)
        mask = self.preallocate_interleaved_mask(mask, bsz, max_gen_len)
        eos_reached = torch.tensor([False] * bsz, device=self.device)

        # do N - 1 iterations to leave room for eot_id
        log_probs_out: List[torch.Tensor] = []
        max_pos = bsz * (force_tokens if force_tokens else max_gen_len - 1)
        for cur_pos in range(0, max_pos, bsz):
            if cur_pos == 0:
                header_ends = torch.cumsum(torch.tensor(header_length, device=self.device), dim=0) - 1
                logits = prefill_logits[:, header_ends]
                if ttft_buffer is not None:
                    first_token_event.record(stream=None)
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
                if log_probs:
                    log_probs_out.append(torch.log(probs.gather(dim=-1, index=next_token.view(1, -1, 1))))
            else:
                next_token = torch.argmax(logits[:, -bsz:], dim=-1).squeeze(0)

            if teacher_force is not None:
                next_token = teacher_force[:, cur_pos // bsz]

            self.node_map[self.cache_len : self.cache_len + bsz][~eos_reached] = \
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)[~eos_reached]
            self.context[self.cache_len : self.cache_len + bsz][~eos_reached] = next_token[~eos_reached]
            self.position_map[self.cache_len : self.cache_len + bsz] = position_ids

            position_ids += 1
            self.cache_len += bsz
            eos_reached |= torch.isin(next_token, self.stop_tokens)

            stop_because_eos = eos_reached.all() and not force_tokens
            stop_because_last_iter = cur_pos == max_pos - bsz
            if (stop_because_eos or stop_because_last_iter) and not stateless:
                # one more forward pass to top off the kv cache
                self.model.forward(
                    tokens=self.context[self.cache_len - bsz : self.cache_len].unsqueeze(0),
                    start_pos=self.cache_len - bsz,
                    mask=mask[:, :self.cache_len],
                    position_ids=position_ids
                )

            if eos_reached.all() and not force_tokens:
                break

        # force decode eot_id for everything that didn't naturally terminate
        if (~eos_reached).any():
            self.node_map[self.cache_len : self.cache_len + bsz][~eos_reached] = \
                torch.arange(self.cur_id, self.cur_id + bsz, device=self.device)[~eos_reached]
            self.context[self.cache_len : self.cache_len + bsz][~eos_reached] = 128009
            self.position_map[self.cache_len : self.cache_len + bsz] = position_ids
            self.cache_len += bsz

        if (~eos_reached).any() and not stateless:
            self.model.forward(
                tokens=self.context[self.cache_len - bsz : self.cache_len].unsqueeze(0),
                start_pos=self.cache_len - bsz,
                mask=mask[:, : self.cache_len],
                position_ids=position_ids + 1
            )

        if debug:
            self.debug_mask(mask[:, : self.cache_len])

        result = self.wrap_outputs(
            self.context[header_end : self.cache_len].view(-1, bsz).t(),
            tasks,
            headers,
            content_prefills
        )
        self.cache_len = header_start if stateless else self.cache_len
        self.cur_id += 0 if stateless else bsz
        if log_probs:
            result['log_probs'] = log_probs_out

        end_event.record(stream=None)
        torch.cuda.synchronize()
        if time_buffer is not None:
            time_buffer.append(start_event.elapsed_time(end_event) / 1000.0)
        if ttft_buffer is not None:
            ttft_buffer.append(start_event.elapsed_time(first_token_event) / 1000.0)

        return result

    def step(self, *args, track_gradients: bool = False, **kwargs) -> StepResult:
        with (nullcontext() if track_gradients else torch.inference_mode()):
                return self._step(*args, **kwargs)

    def train_step(
        self,
        tasks: List[Task],
        target_ids: List[List[int]],
    ) -> Tuple[List[Cached], torch.Tensor]:
        bsz = len(tasks)
        self.add_nodes(tasks)
        mask = self.dynamic_mask(self.cur_id, self.cur_id + bsz)
        headers, header_length, _ = self.build_headers(tasks)
        header_logits = self.prefill(sum(headers, []), header_length, start=self.cur_id, end=self.cur_id+bsz, cached_mask=mask)
        header_ends = torch.cumsum(torch.tensor(header_length, device=self.device), dim=0) - 1
        first_logits = header_logits[:, header_ends]
        target_logits = self.prefill(sum((t[:-1] for t in target_ids), []), [len(t) - 1 for t in target_ids], start=self.cur_id, end=self.cur_id+bsz)
        outputs = [Cached({
            'id': self.cur_id + i,
            'parent_ids': task['parent_ids'],
            'output_tokens': target,
            'output_length': len(target)
        }) for i, (task, target) in enumerate(zip(tasks, target_ids))]
        self.cur_id += bsz
        combined_logits = torch.cat([first_logits, target_logits], dim=1)
        return outputs, combined_logits

    def wrap_outputs(
        self,
        tokens: torch.Tensor,
        tasks: List[Task],
        headers: List[List[int]],
        content_prefills: List[List[int]],
    ) -> StepResult:
        out_tokens: List[List[int]] = []
        out_nodes: List[Cached] = []
        for i, (task, toks, header, content_prefill) in enumerate(
            zip(tasks, tokens.tolist(), headers, content_prefills)
        ):
            stops = []
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    idx = toks.index(stop_token)
                    stops.append(idx)
                except ValueError:
                    pass
            if stops:
                eos_idx = min(stops)
                toks = toks[:eos_idx+1]
                out_tokens.append(content_prefill + toks[:-1])
            else:
                out_tokens.append(content_prefill + toks)
            out_nodes.append(Cached({
                'id': self.cur_id + i,
                'parent_ids': task['parent_ids'],
                'output_tokens': toks[:-1],
                'output_length': len(toks) - 1,
            }))

        return StepResult({
            'tokens': out_tokens,
            'nodes': out_nodes,
            'log_probs': None,
        })

    def add_nodes(self, nodes: Sequence[Node]):
        for i, node in enumerate(nodes):
            for p in node['parent_ids']:
                self.adj[self.cur_id + i, p] = True

    def build_headers(self, tasks: List[Task]) -> Tuple[List[List[int]], List[int], List[List[int]]]:
        headers = []
        header_length = []
        content_prefills = [[] for _ in tasks]
        for i, task in enumerate(tasks):
            role = task['header'][0] + (f':{tag}' if (tag := task['header'][1]) else '')
            header = self.formatter.encode_header({"role": role, "content": ""})
            if (prefill := task.get('prefill')):
                content_prefills[i] = self.tokenizer.encode(prefill, bos=False, eos=False)
                header.extend(content_prefills[i])
            headers.append(header)
            header_length.append(len(header))
        return headers, header_length, content_prefills

    def prefill(
        self,
        _tokens: List[int],
        _length: List[int],
        start: int,
        end: int,
        cached_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = len(_length)
        tokens = torch.tensor(_tokens, device=self.device)
        length = torch.tensor(_length, device=self.device)

        # important invariant: self.cur_id is only mutated AFTER prefilling
        node_par_mask = cached_mask if cached_mask is not None else \
            self.dynamic_mask(self.cur_id, self.cur_id + B)
        node_pos_ids = self.leftmost_position_ids(node_par_mask == 0)

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
            offset += par_length.item()

        self.model.reposition_cache(where, from_pos, to_pos)
        self.position_map[where] = to_pos

    def preallocate_interleaved_mask(self, base_mask: torch.Tensor, bsz: int, max_gen_len: int):
        interleaved_mask = torch.full((bsz, bsz), float("-inf"))
        interleaved_mask.fill_diagonal_(0)
        return torch.hstack([base_mask, interleaved_mask.repeat(1, max_gen_len)])

    def dynamic_mask(self, start_id: int, end_id: int) -> torch.Tensor:
        return torch.where(self.adj[start_id:end_id, self.node_map[:self.cache_len]], 0., float("-inf"))

    def leftmost_position_ids(self, mask: torch.Tensor) -> torch.Tensor:
        return 1 + torch.max(self.position_map[:self.cache_len].expand_as(mask).masked_fill(~mask, 0), dim=1).values

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

def incremental_sequence_with_offset(offsets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    total_length = lengths.sum()
    segment_offsets = torch.repeat_interleave(offsets, lengths)
    position_increments = torch.arange(total_length, device=lengths.device) - torch.repeat_interleave(
        torch.cat([torch.tensor([0], device=lengths.device), lengths.cumsum(0)[:-1]]), lengths
    )
    return segment_offsets + position_increments
