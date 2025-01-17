import os
import time
from typing import List, Dict, Callable
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

import psutil
def log_memory():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

def default_profiler(wait, warmup, active, repeat):
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

def benchmark_workflow(
    workflow_fn: Callable,
    test_cases: List[Dict],
    n_trials: int = 5,
    output_dir: str = "profile"
) -> Dict:
    os.makedirs(Path(output_dir), exist_ok=True)
    results = []
    for i, case in enumerate(test_cases):
        times = []
        outputs = []
        with default_profiler(1, 1, n_trials-2, 1) as prof:
            for trial in range(n_trials):
                print(f"Trial {trial+1}")
                with record_function("full_workflow"):
                    start = time.perf_counter()
                    output = workflow_fn(**case)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                print(f"Finished in {times[-1]}")
                log_memory()
                prof.step()
                outputs.append(output)

        print(f"\nProfile for case {i+1}:")
        prof.export_chrome_trace(f"{output_dir}/trace_{i+1}.json")

        results.append({
            'mean': np.mean(times),
            'std': np.std(times),
            'times': times,
            'profile': prof,
            'outputs': outputs
        })

    return results
