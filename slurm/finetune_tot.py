from llama.workflows.finetune import finetune

finetune(
    data_path='/home/tbai4/llama3/llama/tot_data_2',
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    output_dir='/scratch4/jeisner1/tjbai/checkpoints/tot_2',
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_len=8192,
    checkpoint_freq=50,
    validation_freq=50,
    epochs=2,
    branching_factor=8,
    voters=4,
)
