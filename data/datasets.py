from typing import List
import torch
import pandas as pd
import numpy as np


class CitationCountDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, split: str, 
                 start_year: int, end_year: int, test_year: int,
                 feature_names: List[str], 
                 categorical_names: List[str], 
                 target_names: List[str]):
        super(CitationCountDataset, self).__init__()
        
        self.split = split
        self.path = path
        self.start_year = start_year
        self.end_year = end_year
        if test_year is None:
            self.test_year = self.end_year + 1
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.target_names = target_names
        self.num_features = len(feature_names)
        self.num_targets = len(target_names)
        
        df = pd.read_parquet(path)
        
        if self.split == "train":
            df = df.query(f"year >= {self.start_year} & year <= {self.end_year}")
        elif self.split == "test":
            df = df.query(f"year == {self.test_year}")
            
        df.drop(columns=['citedby_patent_date'])
        df = df.reset_index()
        df = df.dropna(subset = self.feature_names + self.categorical_names + self.target_names)
        
        self.features = df[self.feature_names].values.astype(np.float32)
        self.categorical = {}
        for n in self.categorical_names:
            self.categorical[n] = df[n].values
        # This is to handle the unknown embeddings of all the future years
        if 'year' in self.categorical_names and self.split == "test":
            self.categorical['year'][:] = self.end_year
            
        self.targets = df[self.target_names].values.astype(np.float32)
        
    def __getitem__(self, index: int):
        history = self.features[index]
        target = self.targets[index]
        item = {"history": history, "target": target}
        for c in self.categorical_names:
            item[c] = self.categorical[c][index]
        return item
    
    def __len__(self):
        return len(self.features)