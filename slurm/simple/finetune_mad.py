from llama.workflows.finetune import finetune

finetune(
    task='direct',
    data_path='/home/tbai4/llama3/dumps/simple/from_mad.json',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/direct/from_mad',
    max_seq_len=8*8192,
    epochs=4,
    gradient_accumulation_steps=4,
    checkpoint_freq=25,
    validation_freq=25,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05,
    learning_rate=5e-5,
)
