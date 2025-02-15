import json
from llama.workflows.mad_iterative import load_translations
from bleurt import score

scorer = score.BleurtScorer()

val_split = load_translations('/home/tbai4/llama3/data/commonmt', start=0, end=100)

with open('/home/tbai4/llama3/dumps/mad_iterative/translate_e2e.json', 'r') as f:
    data = json.load(f)

cached = [{
    'candidate': d['decision']['Translation'],
    'reference': item['correct']
} for d, item in zip(data['cached'], val_split) if d.get('decision')]

cached_scores = scorer.score(
    references=[d['reference'] for d in cached],
    candidates=[d['candidate'] for d in cached],
    batch_size=16,
)

baseline = [{
    'candidate': d['decision']['Translation'],
    'reference': item['correct']
} for d, item in zip(data['baseline'], val_split) if d.get('decision')]

baseline_scores = scorer.score(
    references=[d['reference'] for d in baseline],
    candidates=[d['candidate'] for d in baseline],
    batch_size=16,
)

print("Baseline System:")
print(f"- Mean Score: {sum(baseline_scores)/len(baseline_scores):.4f}")

print("Cached System:")
print(f"- Mean Score: {sum(cached_scores)/len(cached_scores):.4f}")
