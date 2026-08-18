"""Run a training experiment from a jetcon config.

Usage:
    python tools/train.py configs/waymo.yaml
    python tools/train.py configs/waymo.yaml provider.scene_idx=555 trainer.max_steps=1000
"""
import argparse

from xsim.trainers import Trainer
from xsim.utils import update_env


def main():
    update_env()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='path to the training config')
    parser.add_argument('overrides', nargs='*',
                        help='config overrides as dotted.key=value (e.g. provider.scene_idx=555)')
    args = parser.parse_args()

    trainer = Trainer(args.config, overrides=args.overrides)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
