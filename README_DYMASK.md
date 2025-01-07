# Workflow Layer:

Each message (represented by (start, end)) can attend over an ordered subset of the previous messages.

Optionally, each message might specify a list of offset locations (of the same length) for those messages.

Messages that attend over the same context but with different offsets _can't_ be generated in parallel.

The orchestrator produces a specification (either ahead of time or during inference?) that can be compiled into a schedule.

Allowing for external code execution, this might look like:

```python
class Task:
    requirements: List[int] # ordered subset of previous messages to attend over
    compact: bool # i'm not sure when we'd ever want to give offsets that aren't fully compacted
    expects: Tuple[Role, str] # role:tag, to prefill the generation prompt. might be problematic if llama isn't pretrained with types.

### example 0: sanity check
system = Message(role='system', 'content'='Respond with a kind greeting.')
user = Message(role='user', 'content'='Hello!')
workflow.insert(chat_format.encode_dialog_prompt([system, user])) # shared context

tasks = [
    Task(requirements=[0, 1], expects=('assistant', None))
    Task(requirements=[0, 1], expects=('assistant', None))
]

resps = workflow.generate([task_1, task_2])

# example 1: static graph for CoT

# example 2: dynamic graph with "moderator"

```

# Model Layer:

At each step we need the attention mask, position ids, and possible rotations.

(Might need to subclass MHA so that rotations occur in some ephemeral buffer that's immediately freed.)

```python
Schedule = List[Step]

class Step:
  initial_attention_mask: torch.Tensor # (1, N, N)
  initial_rotation: torch.Tensor # (1, N, N)
  decoding_positions: List[int]
```
