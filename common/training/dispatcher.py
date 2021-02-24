import os
from typing import Dict
import pathlib
import json
from itertools import product
from train import train
from common.training.hyperparameters import HpList


class Dispatcher:
    def __init__(self, config: Dict, verbose: bool = False):
        values = [list(v) if isinstance(v, HpList) else [v] for v in config.values()]
        self.hyperparams = [dict(zip(config.keys(), v)) for v in product(*values)]
        inp_lists = {k: list(v) for k, v in config.items() if isinstance(v, HpList)}
        values = [v for v in inp_lists.values()]
        variable_values = [dict(zip(inp_lists.keys(), v)) for v in product(*values)]
        folder_names = []
        for d in variable_values:
            folder_names.append(
                ';'.join(['%s=%s' % (key, value) for (key, value) in d.items()])
            )
        self.folder_names = folder_names
        self.verbose = verbose

    def run(self):
        for i, hyperparams in enumerate(self.hyperparams):
            if self.verbose > 0:
                print(f"Dispatching model {i + 1} out of {len(self.hyperparams)}, {self.folder_names[i]}")

            rundir = self.folder_names[i]
            path = os.path.join(hyperparams['expdir'], rundir)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            with open(os.path.join(path, "hyperparams.json"), 'w') as fp:
                json.dump(hyperparams, fp)

            train(hyperparams, rundir=path)