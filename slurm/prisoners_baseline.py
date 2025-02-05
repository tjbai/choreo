from tqdm import tqdm
from llama.workflows.prisoners import prisoners_baseline

llama.model.reshape_cache(2)
llama.model.eval()
payoff = (5, 3, 1, 0)

alice_decisions = []
bob_decisions = []
for seed in tqdm(range(100)):
    baseline_outputs = prisoners_baseline(llama, payoff, seed=seed)
    alice_decisions.append(baseline_outputs['alice_dialog'][-1]['content'])
    bob_decisions.append(baseline_outputs['bob_dialog'][-1]['content'])
    
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
    baseline_outputs = prisoners_baseline(llama, payoff, alice_strategy='always_cooperate', seed=seed)
    alice_decisions.append(baseline_outputs['alice_dialog'][-1]['content'])
    bob_decisions.append(baseline_outputs['bob_dialog'][-1]['content'])
    
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
    baseline_outputs = prisoners_baseline(llama, payoff, alice_strategy='always_defect', seed=seed)
    alice_decisions.append(baseline_outputs['alice_dialog'][-1]['content'])
    bob_decisions.append(baseline_outputs['bob_dialog'][-1]['content'])
    
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
