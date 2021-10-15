import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

my_token = "y1"

class Net(torch.nn.Module):
    def __init__(self,input_features = 16, hidden_features=512, output_features = 16):
        super(Net, self).__init__()
        self.input_features = input_features
        self.h1_features = hidden_features
        self.h2_features = hidden_features*2
        self.output_features = output_features

        self.GCNConv1 = GCNConv(self.input_features,self.h1_features)
        self.GCNConv2 = GCNConv(self.h1_features,self.h2_features)

        self.lstm1 = torch.nn.LSTM( self.input_features+output_features, self.h1_features)
        self.lstm2 = torch.nn.LSTM( self.input_features+output_features, self.h1_features)
        self.L1 = torch.nn.Linear(self.h2_features,self.output_features)
        self.L_temp = torch.nn.Linear(self.input_features, self.output_features)

        print("The token of model is %s"%(my_token))

    def forward(self, node_matrix, graph):
        print(node_matrix.size())
        print(graph.size())        
        x = self.GCNConv1(node_matrix, graph)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.GCNConv2(x, graph)
        
        x = self.L1(x)

        # x_temp = self.L_temp(node_matrix)
        return x