import os
import time
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from models.learners import AbstractModel
from datasets.loaders import EpisodeSampler
from tqdm.auto import tqdm
import re


class Trainer:
    def __init__(self, model: AbstractModel, train_dataset: EpisodeSampler,
                 val_datasets: Dict[str, EpisodeSampler],
                 config: Dict, rundir: str):
        self.model = model
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.config = config
        self.train_losses = dict()
        self.rundir = rundir
        self.tmp_logdir = os.path.join('/tmp/', self.rundir, 'logs')
        self.perm_logdir = os.path.join(self.rundir)
        self.tensorboard = SummaryWriter(self.tmp_logdir, flush_secs=60)

    def sync_logs(self):
        src = re.escape(self.tmp_logdir)
        dst = re.escape(self.perm_logdir)
        cmd = f"rsync -avr {src} {dst}"
        os.system(cmd)

    def update_train_losses(self, train_losses: Dict[str, float], iter: int):
        for k, v in train_losses.items():
            self.train_losses[k] = v
            n = float(iter % self.config['logging_period'] + 1)
            if n == 1.0:
                self.train_losses[k] = v
            else:
                self.train_losses[k] = (self.train_losses[k] * (n - 1) + v) / n

    def train(self):
        last_iteration = self.model.restore()
        inner = tqdm(total=self.config['iterations'], desc='Iteration', initial=last_iteration + 1)

        batch = self.train_dataset.getBatch()

        for iter in range(last_iteration + 1, self.config['iterations']):
            inner.update(1)
            iteration_start = time.time()

            batch = self.train_dataset.getBatch()
            train_losses = self.model.optimize(batch)
            self.update_train_losses(train_losses=train_losses, iter=iter)

            self.model.snapshot(iter=iter, training_losses=None, validation_losses=None)

            inner.set_postfix({key: value for (key, value) in self.train_losses.items()})

            if iter % self.config['evaluation_period'] == 0:
                metrics = self.model.evaluate(self.val_datasets)
                test_logs = {**metrics}
                for k, v in test_logs.items():
                    self.tensorboard.add_scalar(k, v, iter)

            if iter % self.config['logging_period'] == 0:
                self.tensorboard.add_scalar("profiling/iteration", time.time() - iteration_start, iter)
                train_logs = {**train_losses, **(self.model.get_logs())}
                for k, v in train_logs.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iter)
                self.sync_logs()

            self.model.finished_epoch(iter)

        inner.close()
        self.tensorboard.flush()
        self.tensorboard.close()