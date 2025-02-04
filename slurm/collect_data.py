import os
from llama import Llama
from llama.workflows.tot import collect_samples

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

llama = Llama.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
)

samples = collect_samples(
    llama=llama,
    save_dir='/home/tbai4/llama3/dumps',
    n_problems=1000,
    branching_factor=8,
    voters=4,
    temperature=1.0,
    top_p=1.0,
    seed=42,
    math_path='/home/tbai4/llama3/data/MATH',
    split='train',
)

