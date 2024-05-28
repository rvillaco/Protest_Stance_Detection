from torch_geometric.nn.conv import transformer_conv, GCNConv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv, SAGPooling
import torch
import torch.nn.functional as F
from config import *
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm as BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads = ATTN_HEADS)
        self.gbn1 = BatchNorm(hidden_channels*ATTN_HEADS)
        self.pool1 = SAGPooling(hidden_channels*ATTN_HEADS, ratio = 0.8)
        self.conv2 = GATConv((-1, -1), 512, heads = ATTN_HEADS)
        self.gbn2 = BatchNorm(512*ATTN_HEADS)
        #self.pool2 = SAGPooling(512*ATTN_HEADS, ratio = 0.8)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = Linear(round(512*ATTN_HEADS), 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).tanh()
        x, edge_index = self.pool1(x, edge_index)[0:2]
        x = self.dropout(x)
        x = self.conv2(x, edge_index).tanh()
        x = self.gbn2(x)
        x = self.dropout(x)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)

class HomoGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # heads affect dimensions of second layer
        self.gc1 = GATConv((-1, -1),hidden_channels, heads = ATTN_HEADS)
        self.gc2 = GATConv((-1, -1), 1024, heads = ATTN_HEADS)
        self.fc1 = torch.nn.Linear(1024*ATTN_HEADS, 4092)
        self.bn1 = torch.nn.BatchNorm1d(4092)
        self.fc2 = torch.nn.Linear(4092, 1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, out_channels)
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.gc2(x, edge_index).relu()
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = x.log_softmax(dim=-1)
        return x


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


class dumb_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(768, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.linear(x)
        return x.log_softmax(dim=-1)
    
class dumbest_GNN(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = GCNConv(768, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return x.log_softmax(dim=-1)