from llama.workflows.finetune import finetune

finetune(
    task='tot',
    data_path='/home/tbai4/llama3/dumps/tot/tot_math',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/tot/math',
    gradient_accumulation_steps=4,
    max_seq_len=8192,
    checkpoint_freq=50,
    validation_freq=100,
    branching_factor=8,
    voters=4,
    epochs=8,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05,
    learning_rate=5e-5,
)

