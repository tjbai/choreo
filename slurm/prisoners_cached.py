from tqdm import tqdm
from llama.workflows.prisoners import prisoners_cached

workflow.model.reshape_cache(1)
workflow.model.eval()
payoff = (5, 3, 1, 0)

alice_decisions = []
bob_decisions = []
for seed in tqdm(range(100)):
    workflow.reset()
    cached_outputs = prisoners_cached(workflow, payoff, seed=seed)
    alice_decisions.append(workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['tokens']))
    bob_decisions.append(workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['tokens']))
    
print(
    'alice',
    sum(1 for decision in alice_decisions if 'COOPERATE' in decision),
    sum(1 for decision in alice_decisions if 'DEFECT' in decision),
)

print(
    'bob',
    sum(1 for decision in bob_decisions if 'COOPERATE' in decision),
    sum(1 for decision in bob_decisions if 'DEFECT' in decision),
)

alice_decisions = []
bob_decisions = []
for seed in tqdm(range(100)):
    workflow.reset()
    cached_outputs = prisoners_cached(workflow, payoff, alice_strategy='always_cooperate', seed=seed)
    alice_decisions.append(workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['tokens']))
    bob_decisions.append(workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['tokens']))
    
print(
    'alice',
    sum(1 for decision in alice_decisions if 'COOPERATE' in decision),
    sum(1 for decision in alice_decisions if 'DEFECT' in decision),
)

print(
    'bob',
    sum(1 for decision in bob_decisions if 'COOPERATE' in decision),
    sum(1 for decision in bob_decisions if 'DEFECT' in decision),
)

alice_decisions = []
bob_decisions = []
for seed in tqdm(range(100)):
    workflow.reset()
    cached_outputs = prisoners_cached(workflow, payoff, alice_strategy='always_defect', seed=seed)
    alice_decisions.append(workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['tokens']))
    bob_decisions.append(workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['tokens']))
    
print(
    'alice',
    sum(1 for decision in alice_decisions if 'COOPERATE' in decision),
    sum(1 for decision in alice_decisions if 'DEFECT' in decision),
)

print(
    'bob',
    sum(1 for decision in bob_decisions if 'COOPERATE' in decision),
    sum(1 for decision in bob_decisions if 'DEFECT' in decision),
)
