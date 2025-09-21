import json
import argparse
from pathlib import Path
from typing import List


def count_items(paths: List[Path]) -> int:
    total = 0
    for p in paths:
        if p.exists():
            try:
                with open(p) as f:
                    total += len(json.load(f))
            except Exception:
                pass
    return total


def main():
    ap = argparse.ArgumentParser(description='Quick progress view for shard outputs')
    ap.add_argument('--glob', required=True, help='Glob pattern for shard files, e.g., data/baseline_mad_train_shard*-of-5.json')
    args = ap.parse_args()

    paths = sorted(Path('.').glob(args.glob))
    if not paths:
        print('No files matched pattern')
        return
    n = count_items(paths)
    print(f'{len(paths)} shard files matched; {n} items saved total')
    for p in paths:
        try:
            with open(p) as f:
                m = len(json.load(f))
            print(f'- {p}: {m} items')
        except Exception:
            print(f'- {p}: unreadable')


if __name__ == '__main__':
    main()
