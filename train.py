import os
import argparse
import json
from common.training.trainer import Trainer
from typing import Dict
from data import Dataset, DataLoader
import numpy as np
from models import Model


def train(config: Dict, rundir: str):
    
    train_dataset = Dataset(split='train', **config)
    train_loader = DataLoader(train_dataset, drop_last=True, shuffle=True, **config)
    
    test_dataset = Dataset(split='test', **config)
    test_loader = DataLoader(train_dataset, drop_last=False, shuffle=False, **config)
    
    model = Model(config, logdir=rundir)

    trainer = Trainer(model=model,
                      train_dataset=train_sampler_batch,
                      val_datasets={'train': train_loader, 'test': test_loader},
                      config=config, rundir=rundir)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, default='.', help='working directory for this run')
    args = parser.parse_args()

    with open(os.path.join(args.rundir, 'hyperparams.json')) as fp:
        config = json.load(fp)

    train(config, rundir=args.rundir)