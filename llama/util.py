import os
import sys
import time
import json
from pathlib import Path
from typing import Type, Optional, Tuple

import torch
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
    assert 1 <= max_seq_len <= 8192
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
