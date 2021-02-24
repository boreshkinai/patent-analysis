from typing import Dict, Tuple
from abc import abstractmethod, ABC
import os

import torch
import numpy as np

from common.optim import Optimizer, Scheduler
from common.training.snapshotting import SnapshotManager
from models import networks
import inspect
import torch.nn as nn
from models import learners


def logtanh(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.exp(2.0 * x) - 1.0 + 1e-6) - torch.log(torch.exp(2.0 * x) + 1.0)


def logtanh_1(x: torch.Tensor) -> torch.Tensor:
    return np.log(2.0) - torch.log(torch.exp(2.0 * x) + 1.0)


class BackboneList:
    def __init__(self, config):
        self.nets = dict()
        if 'networks' in config:
            for k, c in config['networks'].items():
                self.nets[k] = Backbone(**c)
        else:
            self.nets['common'] = Backbone(**config)


class Backbone(nn.Module):
    def __new__(cls, *args, **kwargs):
        # List available backbones
        backbones = dict()
        for m in inspect.getmembers(networks):
            if inspect.isclass(m[1]):
                backbones[m[0]] = m[1]

        if kwargs["backbone"] in backbones:
            cls = backbones[kwargs["backbone"]]
        else:
            raise Exception(f"Backbone {kwargs['backbone']} is not supported")
        # Filter only arguments expected by the selected class
        expected_args = inspect.getfullargspec(cls)[0][1:]
        expected_kwargs = dict()
        for k, v in kwargs.items():
            if k in expected_args:
                expected_kwargs[k] = v
        return cls(*args, **expected_kwargs)


class AbstractModel(ABC):
    def __init__(self, config: Dict, logdir: str):

        self.nets = BackboneList(config=config)

        self.optimizers_sm = SnapshotManager(snapshot_dir=os.path.join(logdir, 'snapshot_optimizers'),
                                             logging_frequency=config['logging_period'],
                                             snapshot_frequency=config['snapshot_period'])
        self.all_parameters = []
        self.nets_sm = dict()
        for k, n in self.nets.nets.items():
            n.cuda()
            self.nets_sm[k] = SnapshotManager(snapshot_dir=os.path.join(logdir, f"snapshot_model_{k}"),
                                              logging_frequency=config['logging_period'],
                                              snapshot_frequency=config['snapshot_period'])
            self.all_parameters.extend(n.parameters())

        self.optimizer = Optimizer(self.all_parameters, **config)
        self.scheduler = Scheduler(self.optimizer, **config)

        self.config = config
        self.logdir = logdir
        self.losses = dict()

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def optimize(self, batch: Dict[str, torch.Tensor]):
        for k, n in self.nets.nets.items():
            n.train()
            n.zero_grad()
        self.optimizer.zero_grad()

        losses = self.compute_loss(batch=batch)
        total_loss = 0.0
        for k, v in losses.items():
            total_loss = total_loss + v * self.config[f"{k}_scale"]
        total_loss.backward()

        if self.config['clip_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.all_parameters, max_norm=self.config['clip_grad_norm'])
        if self.config['clip_grad_value'] is not None:
            torch.nn.utils.clip_grad_value_(self.all_parameters, clip_value=self.config['clip_grad_value'])
        self.optimizer.step()

        return {k: v.cpu().data.numpy() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, datasets: Dict) -> Dict[str, float]:
        for k, n in self.nets.nets.items():
            n.eval()
        return dict()

    def snapshot(self, iter, training_losses, validation_losses):
        self.optimizers_sm.register(iteration=iter,
                                    training_losses=training_losses,
                                    validation_losses=validation_losses,
                                    model=None,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler)
        for k, n in self.nets.nets.items():
            self.nets_sm[k].register(iteration=iter,
                                     training_losses=training_losses,
                                     validation_losses=validation_losses,
                                     model=n,
                                     optimizer=None,
                                     scheduler=None)

    def restore(self) -> int:
        iter = self.optimizers_sm.restore(model=None, optimizer=self.optimizer, scheduler=self.scheduler)
        for k, n in self.nets.nets.items():
            self.nets_sm[k].restore(model=n, optimizer=None, scheduler=None)
        return iter

    def get_logs(self) -> Dict[str, float]:
        return {"learning_rate": self.scheduler.get_last_lr()[0]}

    def finished_epoch(self, iter):
        self.scheduler.step()


class Forecaster(AbstractModel):
    def __init__(self, config: Dict, logdir: str):
        super().__init__(config=config, logdir=logdir)
        self.losses["cross_entropy"] = nn.CrossEntropyLoss().cuda()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_labels = self.forward(batch)

        predicted_labels = torch.flatten(predicted_labels, start_dim=0, end_dim=1)
        target = torch.flatten(batch['QueryLabel'], start_dim=0, end_dim=1)

        cross_entropy = self.losses["cross_entropy"](input=predicted_labels, target=target)
        return {"cross_entropy": cross_entropy}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = []
        for i in range(inputs['SupportTensor'].shape[0]):
            support_emb = self.nets.nets['features'](inputs['SupportTensor'][i])['features']
            query_emb = self.nets.nets['features'](inputs['QueryTensor'][i])['features']

            prototypes = torch.zeros(size=(5, support_emb.shape[-1])).cuda()
            for l, e in zip(inputs['SupportLabel'][i], support_emb):
                prototypes[l, :] = prototypes[l, :] + e
            prototypes = prototypes / self.config['nSupport']

            distances = query_emb[..., None] - torch.transpose(prototypes, 1, 0)
            distances = - self.config['alpha'] * torch.norm(distances, dim=1)

            output.append(distances)
        return torch.stack(output, dim=0)

    @torch.no_grad()
    def evaluate(self, datasets: Dict) -> Dict[str, float]:
        super(Forecaster, self).evaluate(datasets=datasets)

        NUM_EVALS = 100
        metrics = dict()
        for k, d in datasets.items():
            targets = []
            predictions = []
            for i in range(NUM_EVALS):
                e = d.getBatch()
                p = self.forward(e)

                targets.append(e['QueryLabel'].cpu().numpy())
                predictions.append(torch.argmax(p, dim=-1).cpu().numpy())

            targets = np.concatenate(targets)
            predictions = np.concatenate(predictions)

            metrics[f"{k}/acc"] = (targets == predictions).sum() / targets.size

        return metrics


class Model(AbstractModel):
    def __new__(cls, *args, **kwargs):
        # List available backbones
        models = dict()
        for m in inspect.getmembers(learners):
            if inspect.isclass(m[1]):
                models[m[0]] = m[1]

        config = args[0]

        if config["model"] in models:
            cls = models[config["model"]]
        else:
            raise Exception(f"Model {config['model']} is not supported")
        return cls(*args, **kwargs)
