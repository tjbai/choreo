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
    workflow = Workflow.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialog = [
        {'role': 'system', 'content': 'Respond with a kind greeting.'},
        {'role': 'user', 'content': 'Hello!'}
    ]

    workflow.insert(dialog)

    tasks = [
        {'requirements': [0, 1]},
        {'requirements': [0, 1]},
    ]

    outputs, _ = workflow.step(tasks, max_gen_len, temperature, top_p, log_probs)

if __name__ == '__main__':
    fire.Fire(main)
