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
        {'role': 'system', 'content': 'Echo the user\'s message back.'},
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'user', 'content': 'What is the capital of France?'}
    ]

    system, user_1, user_2 = workflow.insert(dialog)

    outputs, *_ = workflow.step(
        [
            {'requirements': [system, user_1], 'expects': ('assistant', None)},
            {'requirements': [system, user_1], 'expects': ('assistant', None)},
        ]
        max_gen_len, temperature, top_p, log_probs
    )

    outputs, *_ = workflow.step(
        [
            {'requirements': [system, user_2], 'expects': ('assistant', None)},
            {'requirements': [system, user_2], 'expects': ('assistant', None)},
        ]
        max_gen_len, temperature, top_p, log_probs
    )

    outputs, *_ = workflow.step(
        [
            {'requirements': [system, user_1], 'expects': ('assistant', None)},
            {'requirements': [system, user_2], 'expects': ('assistant', None)},
        ]
        max_gen_len, temperature, top_p, log_probs
    )

if __name__ == '__main__':
    fire.Fire(main)
