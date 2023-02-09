"""
Implementation of a Graph Neural Network (GNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from enum import Enum

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim):
        
        super(GNN, self).__init__()

        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)

        self.layer1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.LayerNorm(hidden_dim)

        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = pyg_nn.GCNConv(hidden_dim, hidden_dim)

        self.post_message_passing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = 0.25

    def forward(self, data):
        x, edge_index, batch_size = data.x.float(), data.edge_index, data.batch

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = self.post_message_passing(x)

        x = torch.cat([
            gmp(x, batch_size),
            gap(x, batch_size)], dim=1)

        return F.log_softmax(x, dim=1)

    def loss(self, input, target):
        return F.nll_loss(input, target)
