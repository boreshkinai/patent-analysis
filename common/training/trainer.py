import os
import time
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from models.learners import AbstractModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import re
import torch


class Trainer:
    def __init__(self, model: AbstractModel, train_dataset: DataLoader,
                 val_datasets: Dict[str, DataLoader],
                 config: Dict, rundir: str, tempdir: str = None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.config = config
        self.train_losses = dict()
        self.rundir = rundir
        
        if tempdir is None:
            self.tmp_logdir = os.path.join(self.rundir, 'logs')
        else:
            self.tmp_logdir = os.path.join(tempdir, self.rundir, 'logs')
            
        self.perm_logdir = os.path.join(self.rundir)
        self.tensorboard = SummaryWriter(self.tmp_logdir, flush_secs=30)

    def sync_logs(self):
        src = re.escape(self.tmp_logdir)
        dst = re.escape(self.perm_logdir)
        cmd = f"rsync -avr {src} {dst}"
        os.system(cmd)

    def update_train_losses(self, train_losses: Dict[str, float], iter: int):
        for k, v in train_losses.items():
            self.train_losses[k] = (self.train_losses.get(k, 0.0) * iter + v) / (iter+1)

    def train(self):
        
#         batch_fixed = {'history': torch.Tensor([[0., 0., 0., 0., 0.],
#                                                 [0., 1., 1., 1., 2.]]).cuda(),
#                        'target': torch.Tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                                                [2., 2., 3., 3., 3., 4., 5., 5., 5., 5.]]).cuda()}
        
        
        last_iteration = self.model.restore()        
        for iter in range(last_iteration + 1, self.config['epochs']):
            
            inner = tqdm(total=len(self.train_dataset), desc=f"Epoch {iter}", leave=False)
            for i, batch in enumerate(self.train_dataset):
                inner.update(1)
                iteration_start = time.time()
                
                train_losses = self.model.optimize(batch)
                self.update_train_losses(train_losses=train_losses, iter=i)
                inner.set_postfix({key: value for (key, value) in self.train_losses.items()})
                            
            self.model.snapshot(iter=iter, training_losses=None, validation_losses=None)
            
            if iter % self.config['evaluation_period'] == 0:
                metrics = self.model.evaluate(self.val_datasets)
                test_logs = {**metrics}
                for k, v in test_logs.items():
                    self.tensorboard.add_scalar(k, v, iter)

            if iter % self.config['logging_period'] == 0:
                self.tensorboard.add_scalar("profiling/iteration", time.time() - iteration_start, iter)
                train_logs = {**self.train_losses, **(self.model.get_logs())}
                for k, v in train_logs.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iter)
#                 self.sync_logs()

            self.model.finished_epoch(iter)
            inner.close()

        inner.close()
        self.tensorboard.flush()
        self.tensorboard.close()