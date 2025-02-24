from llama.workflows.finetune import finetune

finetune(
    task='prisoners',
    data_path='/home/tbai4/llama3/dumps/prisoners/prisoners_data',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect',
    max_seq_len=8*8192,
    steps=125,
    gradient_accumulation_steps=4,
    checkpoint_freq=25,
    validation_freq=25,
    lora_alpha=32,
    lora_rank=64,
    lora_dropout=0.05,
    learning_rate=5e-5,
    strategy='always_defect',
)

