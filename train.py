import os
import argparse
import json
from common.training.trainer import Trainer
from typing import Dict
from datasets.loaders import BatchSampler
import numpy as np
from models import Model


def train(config: Dict, rundir: str):
    model = Model(config, logdir=rundir)


    trainer = Trainer(model=model,
                      train_dataset=train_sampler_batch,
                      val_datasets={'train': train_sampler_batch, 'test': test_sampler_batch},
                      config=config, rundir=rundir)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, default='.', help='working directory for this run')
    args = parser.parse_args()

    with open(os.path.join(args.rundir, 'hyperparams.json')) as fp:
        config = json.load(fp)

    train(config, rundir=args.rundir)