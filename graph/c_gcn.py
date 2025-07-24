import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class SlotGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SlotGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        return x
