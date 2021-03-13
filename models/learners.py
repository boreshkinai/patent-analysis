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
from tqdm.auto import tqdm


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
        
        for k, v in batch.items():
            batch[k] = v.cuda()

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
        self.losses["mse"] = nn.MSELoss()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.forward(batch)
        mse_loss = self.losses["mse"](input=output['prediction'], target=batch['target'])
        return {"mse": mse_loss}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        naive = inputs['history'][:,[-1]]
        
        categorical = []
        for k, v in inputs.items():
            if k not in ['history', 'target']:
                categorical.append(inputs[k])
                        
        prediction = self.nets.nets['features'](inputs['history'], *categorical)
        prediction['prediction'] = prediction['prediction'] + naive
        return prediction
    
    @torch.no_grad()
    def predict_on_loader(self, loader: torch.utils.data.DataLoader) -> Dict[str, np.ndarray]:
        super(Forecaster, self).evaluate(datasets=None)

        predictions = []
        for i, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc=f"Predict"):
            for kk, vv in batch.items():
                batch[kk] = vv.cuda()

            p = self.forward(batch)
            predictions.append(p['prediction'])

        predictions = torch.cat(predictions, dim=0).cpu()
            
        return predictions

    @torch.no_grad()
    def evaluate(self, datasets: Dict) -> Dict[str, float]:
        super(Forecaster, self).evaluate(datasets=datasets)

        metrics = dict()
        for k, d in datasets.items():
            targets = []
            predictions = []
            for i, batch in tqdm(enumerate(d), total=len(d), leave=False, desc=f"Evaluation on {k}"):
                for kk, vv in batch.items():
                    batch[kk] = vv.cuda()
                    
                p = self.forward(batch)
                targets.append(batch['target'])
                predictions.append(p['prediction'])

            targets = torch.cat(targets, dim=0)
            predictions = torch.cat(predictions, dim=0)
            
            mse = self.losses["mse"](input=predictions, target=targets)
            
            var = self.losses["mse"](input=torch.mean(targets, dim=list(range(len(targets.shape))), keepdims=True), 
                                                      target=targets)
            metrics[f"{k}/mse"] = mse
            metrics[f"{k}/R2"] = 100.0 * (var - mse) / var
            
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
