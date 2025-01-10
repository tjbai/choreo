import os
import fire
from llama import Workflow

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: int = 128,
):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    workflow = Workflow.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=1
    )

    system, q1, q2 = workflow.insert([
        {'role': 'system', 'content': 'Answer the user\'s question.'},
        {'role': 'user', 'content': 'What is the capital of France?'},
        {'role': 'user', 'content': 'What is the capital of Germany?'},
    ])

    tokens, ids, _ = workflow.step(
        tasks=[
            {'requirements': [system, q1], 'expects': ('assistant', None)},
            {'requirements': [system, q1], 'expects': ('assistant', None)},
            {'requirements': [system, q2], 'expects': ('assistant', None)},
            {'requirements': [system, q2], 'expects': ('assistant', None)},
        ],
        max_gen_len=32,
        temperature=1.0,
        top_p=0.9,
        prefill=True,
        seed=1,
    )

    for i, output in enumerate(tokens):
        print(f'example #{i}')
        print(workflow.tokenizer.decode(output))

if __name__ == '__main__':
    fire.Fire(main)
