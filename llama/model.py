# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_scaled_rope: bool = True
    use_spda: bool = True


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    if x.ndim == 4:
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    elif x.ndim == 5:
        assert freqs_cis.shape == (x.shape[2], x.shape[-1])
        shape = [d if i == 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        raise Exception("Bad shape for broadcast")
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def reposition_rotary_emb(
    xk: torch.Tensor,
    from_pos: torch.Tensor,
    to_pos: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    delta = to_pos - from_pos
    freqs_cis = freqs_cis[delta.abs()]
    freqs_cis = torch.where(delta.unsqueeze(-1) < 0, freqs_cis.conj(), freqs_cis)
    freqs_cis = reshape_for_broadcast(freqs_cis, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, cache_k: torch.Tensor, cache_v: torch.Tensor):
        super().__init__()

        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.use_sdpa = args.use_spda

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = cache_k
        self.cache_v = cache_v

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                xq,
                keys,
                values,
                attn_mask=mask,
                is_causal=False # TODO -- move to flash_attn_with_kv_cache
            )
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, cache_k: torch.Tensor, cache_v: torch.Tensor):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, cache_k, cache_v)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    cache_k: torch.Tensor
    cache_v: torch.Tensor

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.is_lora = False

        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.head_dim = params.dim // params.n_heads

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size,
            params.dim,
            init_method=lambda x: x
        )

        self.register_buffer('cache_k', torch.zeros((
            params.n_layers,
            params.max_batch_size,
            params.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )), persistent=False)

        self.register_buffer('cache_v', torch.zeros((
            params.n_layers,
            params.max_batch_size,
            params.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )), persistent=False)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                layer_id,
                params,
                self.cache_k[layer_id],
                self.cache_v[layer_id]
            ) for layer_id in range(params.n_layers)
        ])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim,
            params.vocab_size,
            bias=False,
            init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def convert_to_lora(self, rank=8, alpha=16):
        self.is_lora = True

        for param in self.parameters():
            param.requires_grad = False

        for layer in self.layers:
            layer.attention.wq = LoraLinear(layer.attention.wq, rank, alpha)
            layer.attention.wk = LoraLinear(layer.attention.wk, rank, alpha)
            layer.attention.wv = LoraLinear(layer.attention.wv, rank, alpha)
            layer.attention.wo = LoraLinear(layer.attention.wo, rank, alpha)

    def get_trainable_parameters(self):
        yield from (param for param in self.parameters() if param.requires_grad)

    def get_trainable_param_percentage(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (trainable_params / total_params) * 100 if total_params > 0 else 0

    def set_adapter_state(self, enabled: bool):
        for layer in self.layers:
            for name in ['wq', 'wk', 'wv', 'wo']:
                adapter = getattr(layer.attention, name)
                if hasattr(adapter, 'disable_adapters'):
                    adapter.disable_adapters = not enabled

    # TODO -- for training this would ideally happen in-place without breaking autograd
    def reposition_cache(self, where: torch.Tensor, from_pos: torch.Tensor, to_pos: torch.Tensor):
        self.cache_k[:, :, where, :, :] = reposition_rotary_emb(
            self.cache_k[:, :, where, :, :],
            from_pos,
            to_pos,
            self.freqs_cis
        )

    def reshape_cache(self, new_batch_size: Optional[int] = None):
        tot = self.params.max_batch_size * self.params.max_seq_len
        if new_batch_size is not None:
            if tot % new_batch_size != 0:
                raise Exception("Invalid shape")
            self.params.max_batch_size = new_batch_size
            self.params.max_seq_len = tot // new_batch_size
        self.cache_k = self.cache_k.view(
            self.cache_k.shape[0],
            self.params.max_batch_size,
            self.params.max_seq_len,
            *self.cache_k.shape[3:]
        )
        self.cache_v = self.cache_v.view(
            self.cache_k.shape[0],
            self.params.max_batch_size,
            self.params.max_seq_len,
            *self.cache_k.shape[3:]
        )
        for i, layer in enumerate(self.layers):
            layer.attention.cache_k = self.cache_k[i]
            layer.attention.cache_v = self.cache_v[i]

    def _forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if position_ids is not None:
            assert position_ids.shape == (seqlen,)
            freqs_cis = self.freqs_cis[position_ids]
        else:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if mask is None:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return self.output(self.norm(h)).float()

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        if not self.training:
            with torch.inference_mode():
                return self._forward(tokens, start_pos, mask, position_ids)
        else:
            return self._forward(tokens, start_pos, mask, position_ids)

class LoraLinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base = base_layer
        self.disable_adapters = False

        base_class = type(base_layer)
        if base_class == ColumnParallelLinear:
            in_dim = base_layer.in_features
            out_dim = base_layer.output_size_per_partition
        else:
            in_dim = base_layer.input_size_per_partition
            out_dim = base_layer.out_features

        self.lora_down = base_class(in_dim, rank, bias=False).to(base_layer.weight.device)
        self.lora_up = base_class(rank, out_dim, bias=False).to(base_layer.weight.device)
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        if self.disable_adapters:
            return self.base(x)
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))
