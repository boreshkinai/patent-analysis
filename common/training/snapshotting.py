import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional
import torch as t


class SnapshotManager:
    def __init__(self, snapshot_dir: str, logging_frequency: int, snapshot_frequency: int):
        self.model_snapshot_file = os.path.join(snapshot_dir, 'model')
        self.optimizer_snapshot_file = os.path.join(snapshot_dir, 'optimizer')
        self.scheduler_snapshot_file = os.path.join(snapshot_dir, 'scheduler')
        self.losses_file = os.path.join(snapshot_dir, 'losses')
        self.iteration_file = os.path.join(snapshot_dir, 'iteration')
        self.time_tracking_file = os.path.join(snapshot_dir, 'time')
        self.logging_frequency = max(logging_frequency, 1)
        self.snapshot_frequency = max(snapshot_frequency, 1)
        self.start_time = None
        self.losses = {'training': {}, 'validation': {}}
        self.time_track = {}

        self.reset_time()

    def restore(self, model: Optional[t.nn.Module], optimizer: Optional[t.optim.Optimizer],
                scheduler: Optional[t.optim.lr_scheduler._LRScheduler]) -> int:
        if model is not None and os.path.isfile(self.model_snapshot_file):
            model.load_state_dict(t.load(self.model_snapshot_file))
        if optimizer is not None and os.path.isfile(self.optimizer_snapshot_file):
            optimizer.load_state_dict(t.load(self.optimizer_snapshot_file))
        if scheduler is not None and os.path.isfile(self.scheduler_snapshot_file):
            scheduler.load_state_dict(t.load(self.scheduler_snapshot_file))
        iteration = t.load(self.iteration_file)['iteration'] if os.path.isfile(self.iteration_file) else -1
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)
            training_losses = {k: v for k, v in losses['training'].items() if k <= iteration}
            validation_losses = {k: v for k, v in losses['validation'].items() if k <= iteration}
            # when restoring remove losses which were after the last snapshot
            self.losses = {'training': training_losses, 'validation': validation_losses}
            self.snapshot(self.losses_file, self.losses)
        if os.path.isfile(self.time_tracking_file):
            self.time_track = t.load(self.time_tracking_file)
        return iteration

    def reset_time(self) -> None:
        self.start_time = time.time()

    def register(self,
                 iteration: int,
                 training_losses: Dict,
                 validation_losses: Dict,
                 model: t.nn.Module,
                 optimizer: Optional[t.optim.Optimizer],
                 scheduler: Optional[t.optim.lr_scheduler._LRScheduler]) -> None:
        if iteration == 1 or iteration % self.logging_frequency == 0:
            self.losses['training'][iteration] = training_losses
            self.losses['validation'][iteration] = validation_losses
            self.snapshot(self.losses_file, self.losses)
            self.snapshot(self.losses_file, self.losses)
        if iteration % self.snapshot_frequency == 0:
            if model is not None:
                self.snapshot(self.model_snapshot_file, model.state_dict())
            if optimizer is not None:
                self.snapshot(self.optimizer_snapshot_file, optimizer.state_dict())
            if scheduler is not None:
                self.snapshot(self.scheduler_snapshot_file, scheduler.state_dict())
            self.snapshot(self.iteration_file, {'iteration': iteration})
            if self.start_time is not None:
                self.time_track[iteration] = time.time() - self.start_time
                self.snapshot(self.time_tracking_file, self.time_track)
                self.start_time = time.time()

    @staticmethod
    def snapshot(path: str, data: Dict) -> None:
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(dir=dir_path, delete=False, mode='wb')
        t.save(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        os.rename(temp_file.name, path)