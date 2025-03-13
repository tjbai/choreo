from llama.workflows.finetune import finetune

finetune(
    task='mad',
    data_path='/home/tbai4/llama3/dumps/mad/math_baseline_e2e.json',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/mad',
    max_seq_len=8*8192,
    epochs=8,
    gradient_accumulation_steps=1,
    checkpoint_freq=450,
    validation_freq=450,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05,
    learning_rate=2e-5,
)
