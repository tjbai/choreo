import os
import json
from tqdm import tqdm
from pathlib import Path

from llama import Workflow
from llama.util import find_free_port
from llama.workflows.bsm import load_concepts
from llama.workflows.bsm import bsm_baseline, bsm_cached

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=100,
)

concepts_list = load_concepts(
    data_path='/home/tbai4/llama3/data/commongen/commongen.jsonl',
    split='dev',
)

output_dir = Path('dumps/bsm')
output_dir.mkdir(parents=True, exist_ok=True)

concepts = []
baseline_stories = []
baseline_all_concepts = []
baseline_coverage = []

cached_stories = []
cached_all_concepts = []
cached_coverage = []

cached_compact_stories = []
cached_compact_all_concepts = []
cached_compact_coverage = []

for idx, concept_set in enumerate(tqdm(concepts_list, desc="Evaluating")):
    concepts.append(concept_set)

    # bsm baseline
    workflow.reset()
    try:
        outputs = bsm_baseline(workflow=workflow, concepts=concept_set)
        if outputs is None:
            raise ValueError("Method returned None")

        story = workflow.tokenizer.decode(outputs['merge_tokens'][0])
        concept_present = [concept.lower() in story.lower() for concept in concept_set]
        all_present = all(concept_present)
        coverage = sum(concept_present) / len(concept_set)

        baseline_stories.append(story)
        baseline_all_concepts.append(all_present)
        baseline_coverage.append(coverage)
    except Exception as e:
        print(f"Error in bsm_baseline: {e}")
        baseline_stories.append(f"ERROR: {str(e)}")
        baseline_all_concepts.append(False)
        baseline_coverage.append(0.0)

    # bsm cached
    workflow.reset()
    try:
        outputs = bsm_cached(workflow=workflow, concepts=concept_set)
        if outputs is None:
            raise ValueError("Method returned None")

        story = workflow.tokenizer.decode(outputs['merge_tokens'][0])
        concept_present = [concept.lower() in story.lower() for concept in concept_set]
        all_present = all(concept_present)
        coverage = sum(concept_present) / len(concept_set)

        cached_stories.append(story)
        cached_all_concepts.append(all_present)
        cached_coverage.append(coverage)
    except Exception as e:
        print(f"Error in bsm_cached: {e}")
        cached_stories.append(f"ERROR: {str(e)}")
        cached_all_concepts.append(False)
        cached_coverage.append(0.0)

    # bsm cached compact
    workflow.reset()
    try:
        outputs = bsm_cached(workflow=workflow, concepts=concept_set, compact=True)
        if outputs is None:
            raise ValueError("Method returned None")

        story = workflow.tokenizer.decode(outputs['merge_tokens'][0])
        concept_present = [concept.lower() in story.lower() for concept in concept_set]
        all_present = all(concept_present)
        coverage = sum(concept_present) / len(concept_set)

        cached_compact_stories.append(story)
        cached_compact_all_concepts.append(all_present)
        cached_compact_coverage.append(coverage)
    except Exception as e:
        print(f"Error in compact baseline: {e}")
        cached_compact_stories.append(f"ERROR: {str(e)}")
        cached_compact_all_concepts.append(False)
        cached_compact_coverage.append(0.0)

    if ((idx + 1) % 10) == 0:
        current = idx + 1

        baseline_all_pct = sum(baseline_all_concepts[:current]) / current * 100
        baseline_avg_pct = sum(baseline_coverage[:current]) / current * 100

        cached_all_pct = sum(cached_all_concepts[:current]) / current * 100
        cached_avg_pct = sum(cached_coverage[:current]) / current * 100

        cached_compact_all_pct = sum(cached_compact_all_concepts[:current]) / current * 100
        cached_compact_avg_pct = sum(cached_compact_coverage[:current]) / current * 100

        print(f"\n===== RESULTS ({current} samples) =====")
        print(f"BSM Baseline:         {baseline_all_pct:.2f}% all concepts, {baseline_avg_pct:.2f}% avg coverage")
        print(f"BSM Cached:           {cached_all_pct:.2f}% all concepts, {cached_avg_pct:.2f}% avg coverage")
        print(f"BSM Cached Compact:   {cached_compact_all_pct:.2f}% all concepts, {cached_compact_avg_pct:.2f}% avg coverage")

total = len(concepts)
final_results = {
    'total_samples': total,
    'methods': {
        'bsm_baseline': {
            'all_concepts_pct': sum(baseline_all_concepts) / total * 100,
            'avg_coverage_pct': sum(baseline_coverage) / total * 100
        },
        'bsm_cached': {
            'all_concepts_pct': sum(cached_all_concepts) / total * 100,
            'avg_coverage_pct': sum(cached_coverage) / total * 100
        },
        'bsm_cached_compact': {
            'all_concepts_pct': sum(cached_compact_all_concepts) / total * 100,
            'avg_coverage_pct': sum(cached_compact_coverage) / total * 100
        }
    },
    'raw_data': {
        'concepts': concepts,
        'baseline': {
            'stories': baseline_stories,
            'all_concepts': baseline_all_concepts,
            'coverage': baseline_coverage
        },
        'cached': {
            'stories': cached_stories,
            'all_concepts': cached_all_concepts,
            'coverage': cached_coverage
        },
        'cached_compact': {
            'stories': cached_compact_stories,
            'all_concepts': cached_compact_all_concepts,
            'coverage': cached_compact_coverage
        }
    }
}

with open('/home/tbai4/llama3/dumps/bsm/initial_eval.json', 'w') as f:
    json.dump(final_results, f)

print("\n===== FINAL RESULTS =====")
for method in ['bsm_baseline', 'bsm_cached', 'bsm_cached_compact']:
    all_pct = final_results['methods'][method]['all_concepts_pct']
    avg_pct = final_results['methods'][method]['avg_coverage_pct']
    print(f"{method}: {all_pct:.2f}% all concepts, {avg_pct:.2f}% avg coverage")
