"""Evaluate a trained experiment (metrics on train + test splits).

Loads the experiment's saved config and a checkpoint, rebuilds the pipeline,
and runs evaluation. Results are printed and appended to the experiment's
``metrics.txt``.

Usage:
    python tools/eval.py outputs/waymo_788/2026-07-07_12-37-34
    python tools/eval.py <experiment_dir> --checkpoint <path/to/nodes_XXXXXX.pth>
"""
import os
import re
import argparse

from xsim.trainers import Trainer
from xsim.utils import update_env


def checkpoint_step(path: str) -> int:
    match = re.search(r'nodes_(\d+)\.pth$', os.path.basename(path or ''))
    return int(match.group(1)) if match else 0


def main():
    update_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', help='experiment directory to evaluate')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to load (default: latest in the dir)')
    args = parser.parse_args()

    trainer = Trainer.from_experiment(args.experiment_dir)
    trainer.build_pipeline()
    trainer.checkpointer.load(args.checkpoint)
    trainer.evaluator.evaluate()


if __name__ == '__main__':
    main()
