from llama.workflows.finetune import finetune

finetune(
    task='triviaqa',
    data_path='/home/tbai4/llama3/dumps/triviaqa/qa_n16.json',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/triviaqa',
    max_seq_len=8192,
    epochs=2,
    gradient_accumulation_steps=4,
    checkpoint_freq=50,
    validation_freq=50,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    learning_rate=5e-5,
)
