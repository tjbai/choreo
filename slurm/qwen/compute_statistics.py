#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np

from llama.util import bootstrap_binary

import config

def load_results(workflow, condition):
    """Load results for a specific workflow and condition"""
    result_file = Path(config.DATA_ROOT) / f'results_{condition}_{workflow}.json'
    if not result_file.exists():
        raise FileNotFoundError(f"Results file not found: {result_file}")

    with open(result_file, 'r') as f:
        return json.load(f)

def compute_pairwise_stats(baseline_results, comparison_results):
    """Compute statistical comparison between two conditions"""
    baseline_correct = baseline_results['all_correct']
    comparison_correct = comparison_results['all_correct']

    if len(baseline_correct) != len(comparison_correct):
        raise ValueError("Result arrays must have same length")

    # Use the bootstrap_binary function from llama.util
    stats = bootstrap_binary(baseline_correct, comparison_correct, n_bootstrap=10000)

    return {
        'baseline_accuracy': baseline_results['accuracy'],
        'comparison_accuracy': comparison_results['accuracy'],
        'accuracy_diff': comparison_results['accuracy'] - baseline_results['accuracy'],
        'binomial_p_value': stats['binomial_p_value'],
        'bootstrap_p_value': stats['bootstrap_p_value'],
        'diff_ci_lower': stats['diff_ci'][0],
        'diff_ci_upper': stats['diff_ci'][1],
        'diff_se': stats['diff_se'],
        'n_samples': len(baseline_correct),
        'baseline_correct_count': sum(baseline_correct),
        'comparison_correct_count': sum(comparison_correct)
    }

def format_accuracy_with_ci(accuracy, diff_ci_lower, diff_ci_upper, baseline_accuracy):
    """Format accuracy with confidence interval like in the table: 38.6 (-5.9, +5.1)"""
    diff = accuracy - baseline_accuracy
    ci_range_lower = diff_ci_lower
    ci_range_upper = diff_ci_upper

    if abs(diff) < 0.001:  # This is the baseline
        return f"{accuracy * 100:.1f}"
    else:
        return f"{accuracy * 100:.1f} ({ci_range_lower * 100:+.1f}, {ci_range_upper * 100:+.1f})"

def compute_workflow_statistics(workflow):
    """Compute all pairwise statistics for a workflow"""
    print(f"\nComputing statistics for {workflow.upper()}...")

    # Load all results for this workflow
    try:
        baseline = load_results(workflow, 'baseline')
        choreo = load_results(workflow, 'choreo')
        choreo_ft = load_results(workflow, 'choreo_ft')
        distilled = load_results(workflow, 'distilled')
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

    results = {
        'workflow': workflow,
        'baseline': baseline,
        'choreographed': choreo,
        'choreo_ft': choreo_ft,
        'distilled': distilled
    }

    # Compute pairwise statistics against baseline
    stats = {}
    for condition in ['choreographed', 'choreo_ft', 'distilled']:
        try:
            stats[condition] = compute_pairwise_stats(baseline, results[condition])
            print(f"{condition} vs baseline: "
                  f"p={stats[condition]['bootstrap_p_value']:.4f}, "
                  f"diff={stats[condition]['accuracy_diff']*100:+.1f}%, "
                  f"CI=({stats[condition]['diff_ci_lower']*100:+.1f}, {stats[condition]['diff_ci_upper']*100:+.1f})")
        except Exception as e:
            print(f"Error computing stats for {condition}: {e}")
            stats[condition] = None

    return {
        'workflow': workflow,
        'results': results,
        'statistics': stats
    }

def generate_results_table(workflows=['tot', 'mad', 'madpar']):
    """Generate formatted results table matching the original format"""
    print("\n" + "="*80)
    print("QWEN RESULTS TABLE")
    print("="*80)

    table_data = []

    for workflow in workflows:
        workflow_stats = compute_workflow_statistics(workflow)
        if workflow_stats is None:
            continue

        results = workflow_stats['results']
        stats = workflow_stats['statistics']

        # Extract data for table
        baseline_acc = results['baseline']['accuracy']

        row_data = {
            'workflow': workflow.upper(),
            'baseline': f"{baseline_acc * 100:.1f}",
            'choreographed': format_accuracy_with_ci(
                results['choreographed']['accuracy'],
                stats['choreographed']['diff_ci_lower'] if stats['choreographed'] else 0,
                stats['choreographed']['diff_ci_upper'] if stats['choreographed'] else 0,
                baseline_acc
            ),
            'choreo_ft': format_accuracy_with_ci(
                results['choreo_ft']['accuracy'],
                stats['choreo_ft']['diff_ci_lower'] if stats['choreo_ft'] else 0,
                stats['choreo_ft']['diff_ci_upper'] if stats['choreo_ft'] else 0,
                baseline_acc
            ),
            'distilled': format_accuracy_with_ci(
                results['distilled']['accuracy'],
                stats['distilled']['diff_ci_lower'] if stats['distilled'] else 0,
                stats['distilled']['diff_ci_upper'] if stats['distilled'] else 0,
                baseline_acc
            )
        }

        table_data.append(row_data)

    # Print formatted table
    print(f"{'Workflow':<10} {'Baseline':<8} {'Choreographed':<20} {'Choreo. + FT':<20} {'Distilled':<20}")
    print("-" * 80)

    for row in table_data:
        print(f"{row['workflow']:<10} {row['baseline']:<8} {row['choreographed']:<20} "
              f"{row['choreo_ft']:<20} {row['distilled']:<20}")

    # Save detailed results
    output_file = Path(config.DATA_ROOT) / 'qwen_complete_results.json'
    with open(output_file, 'w') as f:
        json.dump(table_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    return table_data

def main():
    parser = argparse.ArgumentParser(description="Compute statistics for Qwen experiments")
    parser.add_argument('--workflow', choices=['tot', 'mad', 'madpar'],
                        help="Compute stats for specific workflow")
    parser.add_argument('--all', action='store_true',
                        help="Compute stats for all workflows and generate table")
    args = parser.parse_args()

    if args.all:
        generate_results_table(['tot', 'mad', 'madpar'])
    elif args.workflow:
        compute_workflow_statistics(args.workflow)
    else:
        print("Use --workflow <name> or --all")

if __name__ == "__main__":
    main()