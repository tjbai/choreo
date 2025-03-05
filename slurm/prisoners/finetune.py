import fire
from typing import Optional
from llama.workflows.finetune import finetune

def main(strategy: Optional[str]):
    finetune(
        task='prisoners',
        data_path='/home/tbai4/llama3/dumps/prisoners/prisoners_data',
        ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
        tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
        output_dir=f'/scratch4/jeisner1/tjbai/checkpoints/prisoners/{strategy if strategy else 'baseline'}',
        max_seq_len=8*8192,
        steps=200,
        gradient_accumulation_steps=4,
        checkpoint_freq=50,
        validation_freq=50,
        lora_rank=64,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=5e-5,
        strategy=strategy,
    )

if __name__ == '__main__':
    fire.Fire(main)
