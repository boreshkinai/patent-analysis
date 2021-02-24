import torch
import inspect


class Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __new__(cls, *args, **kwargs):

        if "scheduler" not in kwargs.keys():
            return None

        # List schedulers available in pytorch
        torch_schedulers = dict()
        for m in inspect.getmembers(torch.optim.lr_scheduler):
            if inspect.isclass(m[1]):
                torch_schedulers[m[0]] = m[1]
        # Here we list schedulers that are available from our own packages
        custom_schedulers = dict()
        if kwargs["scheduler"] in torch_schedulers:
            cls = torch_schedulers[kwargs["scheduler"]]
        elif kwargs["scheduler"] in custom_schedulers:
            cls = custom_schedulers[kwargs["scheduler"]]
        else:
            raise Exception(f"Scheduler {kwargs['scheduler']} is not supported")
        # Filter only arguments expected by the selected class
        expected_args = inspect.getfullargspec(cls)[0][1:]
        expected_args.remove('optimizer')
        expected_kwargs = dict()
        for k,v in kwargs.items():
            if k in expected_args:
                expected_kwargs[k] = v
        return cls(*args, **expected_kwargs)


class Optimizer(torch.optim.Optimizer):
    def __new__(cls, *args, **kwargs):

        if "optimizer" not in kwargs.keys():
            return None

        # List optimizers available in pytorch
        torch_optimizers = dict()
        for m in inspect.getmembers(torch.optim):
            if inspect.isclass(m[1]):
                torch_optimizers[m[0]] = m[1]
        # Here we list optimizers that are available from our own packages
        custom_optimizers = dict()
        if kwargs["optimizer"] in torch_optimizers:
            cls = torch_optimizers[kwargs["optimizer"]]
        elif kwargs["optimizer"] in custom_optimizers:
            cls = custom_optimizers[kwargs["optimizer"]]
        else:
            raise Exception(f"Optimizer {kwargs['optimizer']} is not supported")
        # Filter only arguments expected by the selected class
        expected_args = inspect.getfullargspec(cls)[0][1:]
        expected_kwargs = dict()
        for k,v in kwargs.items():
            if k in expected_args:
                expected_kwargs[k] = v
        return cls(*args, **expected_kwargs)