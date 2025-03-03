import os
import sys
import time
import json
import warnings
import socket
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import numpy as np
from tqdm import tqdm
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
) -> Tuple[Transformer, Tokenizer]:
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
    if not (1 <= max_seq_len <= 8192):
        warnings.warn(f"{max_seq_len} does not lie within [1, 8192]")
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
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

    assert model_parallel_size == len(checkpoints), \
        f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )

    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert model_args.vocab_size == tokenizer.n_words

    if torch.cuda.is_bf16_supported():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)

    model = Transformer(model_args).cuda()
    model.load_state_dict(checkpoint, strict=strict)
    if use_lora:
        print("Converting to LoRA")
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
        'result': min(p_value, 1),
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'model1_correct': model1_only,
        'model2_correct': model2_only,
        'n_discordant': n
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
        'bootstrap_p_value': np.mean(np.abs(bootstrap_diff) >= np.abs(observed_diff)),
        'baseline_mean': baseline_arr.mean(),
        'cached_mean': our_arr.mean(),
        'diff_mean': observed_diff,
        'diff_ci': np.percentile(bootstrap_diff, [2.5, 97.5]),
        'diff_se': np.std(bootstrap_diff),
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

    diff_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    diff_se = np.std(bootstrap_means)

    return {'mean_diff': mean_diff, 'diff_ci': diff_ci, 'diff_se': diff_se}
