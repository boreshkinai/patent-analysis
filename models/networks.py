import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FCBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int):
        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.relu_layers = [torch.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(torch.nn.Dropout(p=dropout))
            self.relu_layers.append(torch.nn.Identity())
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [torch.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

        self.forward_projection = torch.nn.Linear(layer_width, size_out, bias=False)
        self.backward_projection = torch.nn.Linear(size_in, layer_width, bias=False)
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.relu_layers = torch.nn.ModuleList(self.relu_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = torch.relu(h + self.backward_projection(x))
        return b, f


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(num_features))
        self.b_2 = torch.nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FCBlockNorm(FCBlock):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int):
        super(FCBlockNorm, self).__init__(num_layers=num_layers, layer_width=layer_width,
                                          dropout=dropout, size_in=size_in, size_out=size_out)
        self.norm = LayerNorm(num_features=size_in)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = torch.relu(h + self.backward_projection(x))
        return b, f


class Embedding(torch.nn.Module):
    """Implementation of embedding using one hot encoded input and fully connected layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.projection = torch.nn.Linear(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e_ohe = F.one_hot(e, num_classes=self.num_embeddings).float()
        return self.projection(e_ohe)


class FcrWnEmbeddings(torch.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(self,
                 num_blocks: int, num_layers: int, layer_width,
                 dropout: float,
                 size_in: int, size_out: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int):
        super().__init__()

        self.embedding_num = embedding_num
        self.layer_width = layer_width
        
        if self.embedding_num > 0:
            self.embeddings = [Embedding(num_embeddings=embedding_size, 
                                         embedding_dim=embedding_dim) for _ in range(embedding_num)]
        else:
            self.embeddings = []

        self.encoder_blocks = [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout,
                                           size_in=size_in + embedding_dim * embedding_num, size_out=size_out)] + \
                              [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout,
                                           size_in=layer_width, size_out=size_out) for _ in range(num_blocks - 1)]

        self.model = torch.nn.ModuleList(self.encoder_blocks + self.embeddings)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        x the continuous input
        e the categorical inputs
        """

        ee = [x]
        if self.embedding_num > 0:
            for i, v in enumerate(args):
                ee.append(self.embeddings[i](v))
        backcast = torch.cat(ee, dim=-1)
        
        forecast = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, f = block(backcast)
            forecast = forecast + f

        return {'prediction': forecast}


