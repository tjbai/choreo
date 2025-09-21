import argparse
from typing import Optional

from .collect_baseline import collect_baseline_data as baseline_collect, merge_shards as baseline_merge
from .collect_choreo import collect_choreo_data as choreo_collect, merge_shards as choreo_merge
from .eval_sharded import eval_choreo_ft, eval_distilled, merge as eval_merge
from .collect_direct import collect_direct_baseline, merge as direct_merge
from .status import main as status_main


def main():
    ap = argparse.ArgumentParser(description='Qwen experiments control (qwenctl)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('baseline')
    sb = b.add_subparsers(dest='action', required=True)
    bc = sb.add_parser('collect')
    bc.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    bc.add_argument('--split', choices=['train', 'test', 'both'], default='both')
    bc.add_argument('--shard-index', type=int, default=None)
    bc.add_argument('--num-shards', type=int, default=None)
    bc.add_argument('--start', type=int, default=0)
    bc.add_argument('--limit', type=int, default=None)
    bc.add_argument('--resume', action='store_true')
    bc.add_argument('--output-dir', type=str, default=None)

    bm = sb.add_parser('merge')
    bm.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    bm.add_argument('--split', choices=['train', 'test'], required=True)
    bm.add_argument('--num-shards', type=int, required=True)
    bm.add_argument('--output-dir', type=str, default=None)

    # choreo
    c = sub.add_parser('choreo')
    sc = c.add_subparsers(dest='action', required=True)
    cc = sc.add_parser('collect')
    cc.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    cc.add_argument('--shard-index', type=int, default=None)
    cc.add_argument('--num-shards', type=int, default=None)
    cc.add_argument('--start', type=int, default=0)
    cc.add_argument('--limit', type=int, default=None)
    cc.add_argument('--resume', action='store_true')
    cc.add_argument('--output-dir', type=str, default=None)

    cm = sc.add_parser('merge')
    cm.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    cm.add_argument('--num-shards', type=int, required=True)
    cm.add_argument('--output-dir', type=str, default=None)

    # eval
    e = sub.add_parser('eval')
    se = e.add_subparsers(dest='action', required=True)
    ec = se.add_parser('choreo_ft')
    ec.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    ec.add_argument('--checkpoint-dir', type=str, required=True)
    ec.add_argument('--shard-index', type=int, default=None)
    ec.add_argument('--num-shards', type=int, default=None)
    ec.add_argument('--start', type=int, default=0)
    ec.add_argument('--limit', type=int, default=None)
    ec.add_argument('--resume', action='store_true')
    ec.add_argument('--output-dir', type=str, default=None)

    ed = se.add_parser('distilled')
    ed.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    ed.add_argument('--checkpoint-dir', type=str, required=True)
    ed.add_argument('--shard-index', type=int, default=None)
    ed.add_argument('--num-shards', type=int, default=None)
    ed.add_argument('--start', type=int, default=0)
    ed.add_argument('--limit', type=int, default=None)
    ed.add_argument('--resume', action='store_true')
    ed.add_argument('--output-dir', type=str, default=None)

    em = se.add_parser('merge')
    em.add_argument('--kind', choices=['choreo_ft', 'distilled'], required=True)
    em.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    em.add_argument('--num-shards', type=int, required=True)
    em.add_argument('--output-dir', type=str, default=None)

    # direct baseline
    d = sub.add_parser('direct')
    sd = d.add_subparsers(dest='action', required=True)
    dc = sd.add_parser('collect')
    dc.add_argument('--shard-index', type=int, default=None)
    dc.add_argument('--num-shards', type=int, default=None)
    dc.add_argument('--start', type=int, default=0)
    dc.add_argument('--limit', type=int, default=None)
    dc.add_argument('--resume', action='store_true')
    dc.add_argument('--output-dir', type=str, default=None)

    dm = sd.add_parser('merge')
    dm.add_argument('--num-shards', type=int, required=True)
    dm.add_argument('--output-dir', type=str, default=None)

    # status
    s = sub.add_parser('status')
    s.add_argument('--glob', required=True)

    args = ap.parse_args()

    if args.cmd == 'baseline':
        if args.action == 'collect':
            baseline_collect(
                args.workflow,
                split=args.split,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
            )
        else:
            baseline_merge(args.workflow, args.split, args.num_shards, output_dir=args.output_dir)

    elif args.cmd == 'choreo':
        if args.action == 'collect':
            choreo_collect(
                args.workflow,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
            )
        else:
            choreo_merge(args.workflow, args.num_shards, output_dir=args.output_dir)

    elif args.cmd == 'eval':
        if args.action == 'choreo_ft':
            eval_choreo_ft(
                args.workflow,
                args.checkpoint_dir,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
            )
        elif args.action == 'distilled':
            eval_distilled(
                args.workflow,
                args.checkpoint_dir,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
            )
        else:
            eval_merge(args.kind, args.workflow, args.num_shards, output_dir=args.output_dir)

    elif args.cmd == 'direct':
        if args.action == 'collect':
            collect_direct_baseline(
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
            )
        else:
            direct_merge(args.num_shards, output_dir=args.output_dir)

    else:
        status_main()


if __name__ == '__main__':
    main()
