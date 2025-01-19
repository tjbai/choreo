import os
import json
from typing import List, Dict, Any, Union

from .benchmark import benchmark
from .tot import tot_cached, tot_baseline

def generate_benchmark_cases(path: Path, branching_factors: int, voters: int,) -> List[Dict[str, Any]]:
    """
    Creates a list of test cases for the TOT approach, sweeping
    over different branching_factors and voter counts. Each
    problem is tested with each combination.

    method can be 'cached' or 'baseline' to pick tot_cached vs tot_baseline.
    """
    cases = []
    for problem in problems:
        for bf in branching_factors:
            for v in voters_list:
                # Expand or refine with your own parameters:
                cases.append({
                    "method": method,
                    "problem": problem,
                    "branching_factor": bf,
                    "voters": v,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_gen_len_proposal": max_gen_len_proposal,
                    "max_gen_len_vote": max_gen_len_vote,
                    "max_gen_len_final": max_gen_len_final,
                    "seed": seed
                })
    return cases

def run_tot_experiment(
    method: str,
    problem: str,
    branching_factor: int,
    voters: int,
    temperature: float,
    top_p: float,
    max_gen_len_proposal: int,
    max_gen_len_vote: int,
    max_gen_len_final: int,
    seed: int,
    workflow_or_llama: Any
) -> Dict[str, Any]:
    """
    Single invocation that runs either tot_cached or tot_baseline on a given problem,
    returning a dictionary with performance metrics + output stats.

    `workflow_or_llama` is whichever object you need:
      - If method='cached', you pass a Workflow instance, e.g. 'workflow=...'
      - If method='baseline', you pass a Llama instance, e.g. 'llama=...'

    Returns a dictionary of results (e.g. chosen proposal distribution, times, etc.).
    """

    # The TOT calls can be instrumented with partial-step timers if desired.
    # But for now, let's just do the entire TOT in one function call
    # (which is what the 'benchmark(...)' harness times anyway).

    if method == "cached":
        # TOT with prompt caching
        output = tot_cached(
            workflow=workflow_or_llama,
            problem=problem,
            branching_factor=branching_factor,
            voters=voters
        )
    elif method == "baseline":
        # TOT with baseline approach
        output = tot_baseline(
            llama=workflow_or_llama,
            problem=problem,
            branching_factor=branching_factor,
            voters=voters,
            temperature=temperature,
            top_p=top_p,
            max_gen_len_proposal=max_gen_len_proposal,
            max_gen_len_vote=max_gen_len_vote,
            max_gen_len_final=max_gen_len_final,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Optionally parse some meta-information from `output`
    # e.g. distribution of which proposal was chosen
    chosen = output['votes'] if 'votes' in output else []
    # E.g. how many times each proposal got chosen:
    from collections import Counter
    count_chosen = dict(Counter(chosen))

    # Potentially measure total tokens across proposals, votes, final:
    num_proposal_tokens = sum(len(pt) for pt in output.get('proposal_tokens', []))
    num_vote_tokens = sum(len(vt) for vt in output.get('vote_tokens', []))
    num_final_tokens = len(output.get('res', [])) if output.get('res', None) else 0

    # Return a dictionary that includes these metrics
    return {
        "chosen_distribution": count_chosen,
        "proposal_tokens": num_proposal_tokens,
        "vote_tokens": num_vote_tokens,
        "final_tokens": num_final_tokens,
        "output_obj": output
    }


def main():
    # ------------------------------------------------
    # 0) Setup: Decide on your problem sets, parameters, etc.
    # ------------------------------------------------
    # TODO: You might read from a JSON file or something more elaborate.
    # For demonstration, let's define a minimal list of problem statements:
    problems = [
        "Find the derivative of x^2 + 3x + 5.",
        "What is the largest prime less than 100?",
    ]
    # Example parameter grids:
    branching_factors = [2, 3]
    voters_list = [1, 2]
    methods = ["cached", "baseline"]
    n_trials = 3  # how many times to run each config
    output_dir = "benchmark_results"

    # We will produce a "workflow" instance or a "llama" instance for each method
    # in a real script, you'd build the model(s) here. For now we put a TODO:
    # e.g. for cached:
    # workflow = Workflow.build(
    #   ckpt_dir=..., tokenizer_path=..., max_seq_len=..., max_batch_size=...,
    #   model_parallel_size=..., seed=42
    # )
    # e.g. for baseline:
    # llama = Llama.build(...)

    # For demonstration, we'll just store placeholders
    workflow_placeholder = None  # TODO: create actual Workflow if needed
    llama_placeholder = None     # TODO: create actual Llama if needed

    # We'll separate the cases for each method
    # (Or unify them into a single set of test args, but let's keep it simple).
    testcases = []
    for method in methods:
        # Build arguments for that method
        # We skip the "workflow_or_llama" param for now, and we'll fill it in the wrapper below.
        param_combinations = generate_benchmark_cases(
            problems=problems,
            branching_factors=branching_factors,
            voters_list=voters_list,
            method=method
        )
        for pc in param_combinations:
            testcases.append(pc)

    # ------------------------------------------------
    # 1) Wrap the run_tot_experiment in a function
    #    suitable for benchmark(...)
    # ------------------------------------------------
    def run_case(**kwargs):
        # We'll choose the correct reference object (workflow or llama) based on method
        if kwargs["method"] == "cached":
            return run_tot_experiment(
                method="cached",
                problem=kwargs["problem"],
                branching_factor=kwargs["branching_factor"],
                voters=kwargs["voters"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_gen_len_proposal=kwargs["max_gen_len_proposal"],
                max_gen_len_vote=kwargs["max_gen_len_vote"],
                max_gen_len_final=kwargs["max_gen_len_final"],
                seed=kwargs["seed"],
                workflow_or_llama=workflow_placeholder
            )
        else:
            return run_tot_experiment(
                method="baseline",
                problem=kwargs["problem"],
                branching_factor=kwargs["branching_factor"],
                voters=kwargs["voters"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_gen_len_proposal=kwargs["max_gen_len_proposal"],
                max_gen_len_vote=kwargs["max_gen_len_vote"],
                max_gen_len_final=kwargs["max_gen_len_final"],
                seed=kwargs["seed"],
                workflow_or_llama=llama_placeholder
            )

    # ------------------------------------------------
    # 2) Perform the benchmark
    # ------------------------------------------------
    results = benchmark(
        fn=run_case,
        args=testcases,
        n_trials=n_trials,
        output_dir=output_dir,
        profile=True  # or False if you don't want profiler traces
    )

    # ------------------------------------------------
    # 3) Store or display results
    # ------------------------------------------------
    # `results` is a List[BenchmarkResult], one entry per test case in the same order as testcases.
    # Let's combine them into a more structured form to eventually save as JSON.
    aggregated = []
    for case, result_dict in zip(testcases, results):
        # Each result_dict has 'mean', 'std', 'times', 'profile', 'outputs'
        # 'outputs' is the list of return values from run_case for each trial.
        # If you want to store the last output, do:
        last_output = result_dict["outputs"][-1] if result_dict["outputs"] else {}

        entry = {
            "method": case["method"],
            "problem": case["problem"],
            "branching_factor": case["branching_factor"],
            "voters": case["voters"],
            "n_trials": n_trials,
            "mean_time": result_dict["mean"],
            "std_time": result_dict["std"],
            # Possibly store some stats from last_output:
            "chosen_distribution": last_output.get("chosen_distribution", {}),
            "proposal_tokens_count": last_output.get("proposal_tokens", 0),
            "vote_tokens_count": last_output.get("vote_tokens", 0),
            "final_tokens_count": last_output.get("final_tokens", 0),
        }
        aggregated.append(entry)

    # Print them or write them to a file
    print("\n===== Benchmark Results =====\n")
    for row in aggregated:
        print(row)

    # Write to a JSON file for future analysis
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "tot_benchmark_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nWrote summary to {outpath}\n")
