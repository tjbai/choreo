import os
import time
from typing import List, Dict, Callable, TypedDict, Optional, Any
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

import psutil
def log_memory():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

def default_profiler(wait, warmup, active, repeat) -> profile:
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

class BenchmarkResult(TypedDict):
    mean: float
    std: float
    times: List[float]
    profile: Optional[profile]
    outputs: List[Any]

def benchmark(
    fn: Callable,
    args: List[Dict],
    n_trials: int = 5,
    output_dir: str = "benchmark",
    profile: bool = False
) -> List[BenchmarkResult]:
    os.makedirs(Path(output_dir), exist_ok=True)
    results = []
    for i, case in enumerate(args):
        times = []
        outputs = []
        if profile:
            with default_profiler(1, 1, n_trials-2, 1) as prof:
                for trial in range(n_trials):
                    with record_function("full_workflow"):
                        start = time.perf_counter()
                        output = fn(**case)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                    log_memory()
                    prof.step()
                    outputs.append(output)

                print(f"\nProfile for case {i+1}:")
                prof.export_chrome_trace(f"{output_dir}/trace_{i+1}.json")
        else:
            prof = None
            for trial in range(n_trials):
                start = time.perf_counter()
                output = fn(**case)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        results.append({
            'mean': np.mean(times),
            'std': np.std(times),
            'times': times,
            'profile': prof,
            'outputs': outputs
        })

    return results
