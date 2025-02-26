import torch

ckpt = torch.load('/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_epoch-0_step-195.pt', weights_only=True)

for weight, param in zip(ckpt['trainable_params'], workflow.model.get_trainable_parameters()):
    param.data.copy_(weight)
