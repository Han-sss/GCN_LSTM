import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self,input_features = 16, hidden_features=512, output_features = 16):
        super(Net, self).__init__()
        self.input_features = input_features
        self.h1_features = hidden_features
        self.h2_features = hidden_features*2
        self.output_features = output_features

        self.GCNConv1 = GCNConv(self.input_features,self.h1_features)
        self.GCNConv2 = GCNConv(self.h1_features,self.h2_features)
        self.L1 = torch.nn.Linear(self.h2_features,self.output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x.size())

        # print(type(edge_index),edge_index.size())
        x = self.GCNConv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.GCNConv2(x,edge_index)
        
        x = self.L1(x)
        return x