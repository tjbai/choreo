import os, sys, time, json, warnings, socket, glob
from typing import Optional, Tuple, List, Dict, Union, Literal
from pathlib import Path

import torch
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer

def load_model_and_tokenizer(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    model_parallel_size: Optional[int] = None,
    seed: int = 1,
    strict: bool = True,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    model_type: Union[Literal['llama'], Literal['qwen']] = 'llama',
) -> Tuple[Transformer, Tokenizer]:
    """
    Build a Llama or Qwen model instance by initializing and loading a checkpoint.
    Auto-detects model type based on checkpoint format.

    Args:
        ckpt_dir (str): Path to the directory containing checkpoint files.
        tokenizer_path (str): Path to the tokenizer file.
        max_seq_len (int): Maximum sequence length for input text.
        max_batch_size (int): Maximum batch size for inference.
        model_parallel_size (Optional[int], optional): Number of model parallel processes.
            If not provided, it's determined from the environment. Defaults to None.

    Returns:
        Tuple[Transformer, Tokenizer]: Loaded model and tokenizer.

    Raises:
        AssertionError: If there are no checkpoint files in the specified directory,
            or if the model parallel size does not match the number of checkpoint files.
    """

    if not (1 <= max_seq_len <= 32768):
        warnings.warn(f"{max_seq_len} does not lie within [1, 32768]")

    assert os.path.isdir(ckpt_dir)
    assert os.path.isfile(tokenizer_path)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(seed)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    start_time = time.time()

    if model_type == "llama":
        llama_pths = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(llama_pths) > 0, f"No Llama checkpoint files found in {ckpt_dir}"
        ckpt_path = llama_pths[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path, model_type="llama")
        assert model_args.vocab_size == tokenizer.n_words

        if torch.cuda.is_bf16_supported():
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.bfloat16)
        else:
            torch.set_default_dtype(torch.float16)

        model = Transformer(model_args).cuda()
        model.load_state_dict(checkpoint, strict=strict)

    elif model_type == "qwen":
        qwen_sfts = sorted(list(Path(ckpt_dir).glob("model*.safetensors")))
        qwen_config = Path(ckpt_dir) / "config.json"

        assert qwen_sfts and qwen_config.exists(), f"No Qwen checkpoint files found in {ckpt_dir}"
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        assert model_parallel_size == 1, "Qwen loader currently supports MP=1 only."

        with open(qwen_config, "r") as f:
            cfg = json.load(f)

        hidden_size = cfg.get("hidden_size", cfg.get("n_embd"))
        n_layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
        n_heads = cfg.get("num_attention_heads", cfg.get("n_head"))
        n_kv_heads = cfg.get("num_key_value_heads", cfg.get("n_kv_head", n_heads))
        norm_eps = cfg.get("rms_norm_eps", 1e-5)
        rope_theta = cfg.get("rope_theta", 10000)
        attention_bias = cfg.get("attention_bias", False)
        tokenizer = Tokenizer(model_path=ckpt_dir, model_type="qwen")

        model_args = ModelArgs(
            dim=int(hidden_size),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            vocab_size=int(tokenizer.n_words),
            norm_eps=float(norm_eps),
            rope_theta=float(rope_theta),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        model_args.model_type = "qwen"
        model_args.attention_bias = attention_bias
        model_args.use_qk_norm = True

        if torch.cuda.is_bf16_supported():
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.bfloat16)
        else:
            torch.set_default_dtype(torch.float16)

        model = Transformer(model_args).cuda()

        state_dict = _load_qwen_to_native_state_dict(ckpt_dir, target_vocab=model_args.vocab_size)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

    if use_lora:
        print("Converting to LoRA")
        assert lora_rank and lora_alpha and (lora_dropout is not None)
        model.convert_to_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port

def mcnemars_test(_model1: List[bool], _model2: List[bool]) -> Dict:
    model1 = np.array(_model1)
    model2 = np.array(_model2)

    both_correct = np.sum((model1 == True) & (model2 == True))
    both_incorrect = np.sum((model1 == False) & (model2 == False))
    model1_only = np.sum((model1 == True) & (model2 == False))
    model2_only = np.sum((model1 == False) & (model2 == True))
    table = [[both_correct, model1_only], [model2_only, both_incorrect]]

    return {
        'result': mcnemar(table, exact=True).pvalue,
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'model1_correct': model1_only,
        'model2_correct': model2_only,
    }

def binomial_test(_model1: List[bool], _model2: List[bool]) -> Dict:
    model1 = np.array(_model1)
    model2 = np.array(_model2)

    both_correct = np.sum((model1 == True) & (model2 == True))
    both_incorrect = np.sum((model1 == False) & (model2 == False))
    model1_only = np.sum((model1 == True) & (model2 == False))
    model2_only = np.sum((model1 == False) & (model2 == True))

    n = model1_only + model2_only
    k = model1_only

    p_value = (
        2 * min(stats.binom.cdf(k, n, 0.5), 1 - stats.binom.cdf(k - 1, n, 0.5))
        if n > 0 else 1
    )

    return {
        'result': float(min(p_value, 1)),
        'both_correct': float(both_correct),
        'both_incorrect': float(both_incorrect),
        'model1_correct': float(model1_only),
        'model2_correct': float(model2_only),
        'n_discordant': float(n)
    }

def load_ckpt(workflow, ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=True)
    for weight, param in zip(ckpt['trainable_params'], workflow.model.get_trainable_parameters()):
        param.data.copy_(weight)

def bootstrap_binary(
    baseline: List[bool],
    ours: List[bool],
    n_bootstrap=1000
):
    n_samples = len(baseline)
    baseline_arr = np.array(baseline)
    our_arr = np.array(ours)
    assert len(baseline) == len(ours)

    observed_diff = our_arr.mean() - baseline_arr.mean()
    bootstrap_diff = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        baseline_sample = baseline_arr[indices]
        our_sample = our_arr[indices]
        bootstrap_diff.append(our_sample.mean() - baseline_sample.mean())

    return {
        'binomial_p_value': binomial_test(baseline, ours),
        'bootstrap_p_value': float(np.mean(np.abs(bootstrap_diff) >= np.abs(observed_diff))),
        'baseline_mean': float(baseline_arr.mean()),
        'cached_mean': float(our_arr.mean()),
        'diff_mean': float(observed_diff),
        'diff_ci': np.percentile(bootstrap_diff, [2.5, 97.5]).tolist(),
        'diff_se': float(np.std(bootstrap_diff)),
    }

def bootstrap_continuous(
    baseline: List[float],
    cached: List[float],
    n_bootstrap=10000
):
    baseline_arr = np.array(baseline)
    cached_arr = np.array(cached)

    observed_diff = cached_arr - baseline_arr
    mean_diff = np.mean(observed_diff)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(baseline_arr)), size=len(baseline_arr), replace=True)
        bootstrap_means.append(np.mean(observed_diff[indices]))

    return {
        'mean_diff': float(mean_diff),
        'diff_ci': np.percentile(bootstrap_means, [2.5, 97.5]).tolist(),
        'diff_se': float(np.std(bootstrap_means))
    }


def _load_qwen_to_native_state_dict(ckpt_dir: str, target_vocab: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Load Qwen HuggingFace-format weights from `ckpt_dir` and map to this repo's
    native Transformer state_dict layout.

    Notes:
    - Assumes MP=1 (no tensor-parallel sharding).
    - Expects files named model*.safetensors (single or sharded); merges shards.
    - Maps common Qwen key names to our module names.
    - Does not depend on `transformers`; uses safetensors if available.
    """
    try:
        from safetensors.torch import load_file as safe_load
    except Exception as e:
        raise RuntimeError(
            "Loading Qwen safetensors requires `safetensors`. Install via `pip install safetensors`."
        ) from e

    # Load one or multiple shards and merge
    shard_paths = sorted(glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No Qwen safetensors found in {ckpt_dir}")

    merged: Dict[str, torch.Tensor] = {}
    for p in shard_paths:
        shard = safe_load(p)
        for k, v in shard.items():
            merged[k] = v if k not in merged else merged[k]  # last one wins (shouldn't duplicate)

    state: Dict[str, torch.Tensor] = {}

    # Embeddings and output
    if 'model.embed_tokens.weight' in merged:
        emb = merged['model.embed_tokens.weight']
        if target_vocab is not None and emb.shape[0] != target_vocab:
            if emb.shape[0] > target_vocab:
                emb = emb[:target_vocab]
            else:
                pad = torch.zeros((target_vocab - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
        state['tok_embeddings.weight'] = emb
    if 'lm_head.weight' in merged:
        out = merged['lm_head.weight']
        if target_vocab is not None and out.shape[0] != target_vocab:
            if out.shape[0] > target_vocab:
                out = out[:target_vocab]
            else:
                pad = torch.zeros((target_vocab - out.shape[0], out.shape[1]), dtype=out.dtype)
                out = torch.cat([out, pad], dim=0)
        state['output.weight'] = out

    # Final norm
    if 'model.norm.weight' in merged:
        state['norm.weight'] = merged['model.norm.weight']

    # Per-layer mappings
    i = 0
    while True:
        base = f"model.layers.{i}"
        if f"{base}.self_attn.q_proj.weight" not in merged:
            break

        # Attention projections
        state[f"layers.{i}.attention.wq.weight"] = merged[f"{base}.self_attn.q_proj.weight"]
        state[f"layers.{i}.attention.wk.weight"] = merged[f"{base}.self_attn.k_proj.weight"]
        state[f"layers.{i}.attention.wv.weight"] = merged[f"{base}.self_attn.v_proj.weight"]
        state[f"layers.{i}.attention.wo.weight"] = merged[f"{base}.self_attn.o_proj.weight"]

        # Q/K normalization weights (Qwen-specific)
        if f"{base}.self_attn.q_norm.weight" in merged:
            state[f"layers.{i}.attention.q_norm.weight"] = merged[f"{base}.self_attn.q_norm.weight"]
        if f"{base}.self_attn.k_norm.weight" in merged:
            state[f"layers.{i}.attention.k_norm.weight"] = merged[f"{base}.self_attn.k_norm.weight"]

        # Norms
        state[f"layers.{i}.attention_norm.weight"] = merged[f"{base}.input_layernorm.weight"]
        # Qwen uses post_attention_layernorm for MLP input
        state[f"layers.{i}.ffn_norm.weight"] = merged[f"{base}.post_attention_layernorm.weight"]

        # MLP (SwiGLU-style)
        state[f"layers.{i}.feed_forward.w1.weight"] = merged[f"{base}.mlp.gate_proj.weight"]
        state[f"layers.{i}.feed_forward.w2.weight"] = merged[f"{base}.mlp.down_proj.weight"]
        state[f"layers.{i}.feed_forward.w3.weight"] = merged[f"{base}.mlp.up_proj.weight"]

        i += 1

    if i == 0:
        raise RuntimeError("Did not find any Qwen layer weights; unexpected checkpoint format.")

    return state
