import fire
import json
from llama.workflows.finetune import finetune

def main(
    task: str,
    ft: bool = False,
    model_type: str = 'qwen',
    ckpt_dir: str = '/scratch4/jeisner1/tjbai/qwen3_8b',
    tokenizer_path: str = '/scratch4/jeisner1/tjbai/qwen3_8b',
    max_seq_len: int = 8*8192,
    epochs: int = 8,
    gradient_accumulation_steps: int = 1,
    checkpoint_freq: int = 450,
    validation_freq: int = 450,
    lora_rank: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-5,
):
    data = []
    for shard_idx in range(5):
        shard_path = f'/scratch4/jeisner1/tjbai/qwen_data/{task}/train/math_shard-{shard_idx}_ft-{ft}.json'
        with open(shard_path) as f:
            data.extend(json.load(f))

    collected_path = f'/scratch4/jeisner1/tjbai/qwen_data/{task}/train/collected_ft-{ft}.json'
    with open(collected_path, 'w') as f:
        json.dump(data, f)

    print(f"Collected {len(data)} samples from shards -> {collected_path}")
    output_dir = f'/scratch4/jeisner1/tjbai/qwen_data/checkpoints/{task}_ft-{ft}'

    finetune(
        task=task,
        model_type=model_type,
        data_path=collected_path,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_freq=checkpoint_freq,
        validation_freq=validation_freq,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
    )

if __name__ == '__main__':
    fire.Fire(main)
