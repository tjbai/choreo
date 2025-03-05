import json
from llama.workflows.mad import load_translations
from comet import download_model, load_from_checkpoint

model_path = download_model('Unbabel/XCOMET-XL')
model = load_from_checkpoint(model_path)

val_split = load_translations('/home/tbai4/llama3/data/commonmt', start=0, end=100)

with open('/home/tbai4/llama3/dumps/mad_iterative/translate_e2e.json', 'r') as f:
    data = json.load(f)

cached = [{
    'mt': d['decision']['Translation'],
    'src': item['chinese'],
    'ref': item['correct']
} for d, item in zip(data['cached'], val_split) if d.get('decision')]

cached_output = model.predict(cached, batch_size=16, gpus=1)

baseline = [{
    'mt': d['decision']['Translation'],
    'src': item['chinese'],
    'ref': item['correct']
} for d, item in zip(data['baseline'], val_split) if d.get('decision')]

baseline_output = model.predict(baseline, batch_size=16, gpus=1)

print("Baseline System:")
print(f"- Mean Score: {sum(baseline_output.scores)/len(baseline_output.scores):.4f}")

print("Cached System:")
print(f"- Mean Score: {sum(cached_output.scores)/len(cached_output.scores):.4f}")
