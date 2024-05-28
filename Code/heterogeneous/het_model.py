from torch_geometric.nn.conv import transformer_conv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv
import torch
import torch.nn.functional as F
from config import *


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads = ATTN_HEADS)
        self.conv2 = GATConv((-1, -1), 512, heads = ATTN_HEADS)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = Linear(512*ATTN_HEADS, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


## simple sageconv gnn
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)
