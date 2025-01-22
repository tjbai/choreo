import os
import time
from pathlib import Path
from typing import (
    Generic,
    List,
    Callable,
    TypedDict,
    Optional,
    TypeVar,
    ParamSpec,
    Dict,
    Any
)

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

T = TypeVar('T')
P = ParamSpec('P')

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

class BenchmarkResult(TypedDict, Generic[T]):
    mean: float
    std: float
    times: List[float]
    profile: Optional[profile]
    outputs: List[T]

def benchmark(
    fn: Callable[P, T],
    args: List[Dict[str, Any]],
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    output_dir: str = "benchmark",
    profile: bool = False
) -> List[BenchmarkResult[T]]:
    os.makedirs(Path(output_dir), exist_ok=True)
    results = []
    for i, case in enumerate(args):
        times = []
        outputs = []
        if profile:
            with default_profiler(wait, warmup, active, 1) as prof:
                for trial in range(wait + warmup + active):
                    with record_function("full_workflow"):
                        start = time.perf_counter()
                        output = fn(**case)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                    log_memory()
                    prof.step()
                    outputs.append(output)
                print(f"\nFinished {i+1}")
                prof.export_chrome_trace(f"{output_dir}/trace_{i+1}.json")
        else:
            prof = None
            for trial in range(wait + warmup + active):
                start = time.perf_counter()
                output = fn(**case)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                outputs.append(output)

        results.append(BenchmarkResult({
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'times': times,
            'profile': prof,
            'outputs': outputs
        }))

    return results
