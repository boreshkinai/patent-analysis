import torch
import inspect
from data import datasets
from data import loaders


class DataLoader(torch.utils.data.DataLoader):
    def __new__(cls, *args, **kwargs):

        if "loader" not in kwargs.keys():
            return None

        # List schedulers available in pytorch
        available_loaders = dict()
        for m in inspect.getmembers(torch.utils.data):
            if inspect.isclass(m[1]):
                available_loaders[m[0]] = m[1]
        # Here we list schedulers that are available from our own packages
        for m in inspect.getmembers(loaders):
            if inspect.isclass(m[1]):
                available_loaders[m[0]] = m[1]
                
        if kwargs["loader"] in available_loaders:
            cls = available_loaders[kwargs["loader"]]
        else:
            raise Exception(f"Dataset {kwargs['loader']} is not supported")
        # Filter only arguments expected by the selected class
        expected_args = inspect.getfullargspec(cls.__init__)[0][1:]
        expected_args.remove('dataset')
        expected_kwargs = dict()
        for k,v in kwargs.items():
            if k in expected_args:
                expected_kwargs[k] = v
                
        return cls(*args, **expected_kwargs)



class Dataset(torch.utils.data.Dataset):
    def __new__(cls, *args, **kwargs):

        if "dataset" not in kwargs.keys():
            return None

        # List schedulers available in pytorch
        torch_datasets = dict()
        # Here we list schedulers that are available from our own packages
        custom_datasets = dict()
        for m in inspect.getmembers(datasets):
            if inspect.isclass(m[1]):
                custom_datasets[m[0]] = m[1]
                
        if kwargs["dataset"] in torch_datasets:
            cls = torch_datasets[kwargs["dataset"]]
        elif kwargs["dataset"] in custom_datasets:
            cls = custom_datasets[kwargs["dataset"]]
        else:
            raise Exception(f"Dataset {kwargs['dataset']} is not supported")
        # Filter only arguments expected by the selected class
        expected_args = inspect.getfullargspec(cls.__init__)[0][1:]
        expected_kwargs = dict()
        for k,v in kwargs.items():
            if k in expected_args:
                expected_kwargs[k] = v
                
        return cls(*args, **expected_kwargs)