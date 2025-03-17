import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from llama import Workflow
from llama.util import find_free_port, load_ckpt
from llama.workflows.bsm import load_concepts
from llama.workflows.bsm import bsm_cached

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

BASE_MODEL_PATH = '/scratch4/jeisner1/tjbai/llama_8b'
TOKENIZER_PATH = '/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model'
CHECKPOINTS_DIR = '/scratch4/jeisner1/tjbai/checkpoints/bsm'
CONCEPTS_PATH = '/home/tbai4/llama3/data/commongen/commongen.jsonl'
OUTPUT_DIR = Path('dumps/bsm/checkpoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

workflow = Workflow.build(
    ckpt_dir=BASE_MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05
)
workflow.model.eval()

concepts_list = load_concepts(data_path=CONCEPTS_PATH, split='dev')

all_checkpoint_results = {}

for ckpt_file in sorted(os.listdir(CHECKPOINTS_DIR)):
    if not ckpt_file.endswith('.pt'):
        continue

    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_file)
    ckpt_name = os.path.splitext(ckpt_file)[0]
    print(f"\nEvaluating checkpoint: {ckpt_name}")

    load_ckpt(workflow, ckpt_path)

    stories = []
    all_concepts_covered = []
    coverages = []
    group1_coverages = []
    group2_coverages = []

    for concept_set in tqdm(concepts_list, desc=f"Evaluating {ckpt_name}"):
        workflow.reset()
        try:
            outputs = bsm_cached(workflow=workflow, concepts=concept_set)
            if outputs is None:
                raise ValueError("Method returned None")

            story = workflow.tokenizer.decode(outputs['merge_tokens'][0])

            group1_concepts = outputs['concept_groups'][0]
            group2_concepts = outputs['concept_groups'][1]

            concept_present = [concept.lower() in story.lower() for concept in concept_set]
            all_present = all(concept_present)
            coverage = sum(concept_present) / len(concept_set)

            group1_present = [concept.lower() in story.lower() for concept in group1_concepts]
            group1_coverage = sum(group1_present) / len(group1_concepts) if group1_concepts else 0

            group2_present = [concept.lower() in story.lower() for concept in group2_concepts]
            group2_coverage = sum(group2_present) / len(group2_concepts) if group2_concepts else 0

            stories.append(story)
            all_concepts_covered.append(all_present)
            coverages.append(coverage)
            group1_coverages.append(group1_coverage)
            group2_coverages.append(group2_coverage)

        except Exception as e:
            print(f"Error in bsm_cached: {e}")
            stories.append(f"ERROR: {str(e)}")
            all_concepts_covered.append(False)
            coverages.append(0.0)
            group1_coverages.append(0.0)
            group2_coverages.append(0.0)

    total = len(all_concepts_covered)
    stats = {
        'all_concepts_pct': sum(all_concepts_covered) / total * 100,
        'avg_coverage_pct': sum(coverages) / total * 100,
        'avg_coverage_std': np.std(coverages) * 100,
        'group1_coverage_pct': sum(group1_coverages) / total * 100,
        'group1_coverage_std': np.std(group1_coverages) * 100,
        'group2_coverage_pct': sum(group2_coverages) / total * 100,
        'group2_coverage_std': np.std(group2_coverages) * 100
    }

    print(f"\n===== RESULTS FOR {ckpt_name} =====")
    print(f"All concepts: {stats['all_concepts_pct']:.2f}%, "
          f"Avg coverage: {stats['avg_coverage_pct']:.2f}% ± {stats['avg_coverage_std']:.2f}%")
    print(f"Group 1: {stats['group1_coverage_pct']:.2f}% ± {stats['group1_coverage_std']:.2f}%")
    print(f"Group 2: {stats['group2_coverage_pct']:.2f}% ± {stats['group2_coverage_std']:.2f}%")

    checkpoint_results = {
        'checkpoint': ckpt_name,
        'stats': stats,
        'raw_data': {
            'stories': stories,
            'all_concepts': all_concepts_covered,
            'coverage': coverages,
            'group1_coverage': group1_coverages,
            'group2_coverage': group2_coverages
        }
    }

    all_checkpoint_results[ckpt_name] = checkpoint_results

    with open(OUTPUT_DIR / "all_checkpoint_results.json", 'w') as f:
        json.dump(all_checkpoint_results, f, indent=2)

print(f"\nAll checkpoint evaluations completed and saved to {OUTPUT_DIR}")
