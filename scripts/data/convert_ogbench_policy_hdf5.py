#!/usr/bin/env python
"""Convert OGBench Isaac Lab policy HDF5 → compact SWM HDF5 (pixels + action).

Full 40k conversion rewrites ~340 GB of bird-view pixels. Prefer the streaming
``ogbench_policy`` format for training on the full corpus. Use this script for
small smoke subsets or when an on-disk SWM layout is required.

Example::

    uv run python scripts/data/convert_ogbench_policy_hdf5.py \\
        --source /root/ogbench_isaaclab/data/trajectory-generations/train-episodes-40000/shards/shard-000.hdf5 \\
        --dest /tmp/ogb_cube_swm_smoke.h5 \\
        --max-episodes 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_worldmodel.data.formats.ogbench_policy import convert_policy_hdf5_to_swm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Policy HDF5 path')
    p.add_argument('--dest', required=True, help='Output SWM .h5 path')
    p.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Optional cap on number of subtrajectories to convert',
    )
    p.add_argument(
        '--action-horizon-idx',
        type=int,
        default=0,
        help='Receding-horizon index for the transition action (default 0)',
    )
    p.add_argument(
        '--mode',
        choices=('append', 'overwrite', 'error'),
        default='overwrite',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = convert_policy_hdf5_to_swm(
        args.source,
        args.dest,
        action_horizon_idx=args.action_horizon_idx,
        max_episodes=args.max_episodes,
        mode=args.mode,
    )
    print(f'Wrote SWM dataset to {Path(dest).resolve()}')


if __name__ == '__main__':
    main()
