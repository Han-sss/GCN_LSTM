from operator import mod
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

import GCN_model
import GCN_help_function as ghf

epoch_num = 20
my_token = "a1"
suffix = "v1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='/tmp/Cora', name='Cora')
model = GCN_model.Net(input_features=dataset.num_node_features,output_features=dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("the token of now is %s"%(my_token))

model.train()
acc_best = 0
for epoch in range(epoch_num):
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.CrossEntropyLoss()
    loss_res = loss(output[data.train_mask], data.y[data.train_mask])
    loss_res.backward()
    print("the loss of epoch %d is: %f"%(epoch,loss_res.item()))
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            _, pred = model(data).max(dim=1)
            correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            acc = correct / data.val_mask.sum().item()
            if acc > acc_best:
                acc_best = acc
                torch.save(model.state_dict(),'GCN_model_params_%s.pth'%(suffix))
                print('Acc of epoch {}: {:.4f}'.format(epoch,acc))
                
                correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                acc_test = correct / data.test_mask.sum().item()
                print('Acc: {:.4f}'.format(acc_test))

model.load_state_dict(torch.load('GCN_model_params_%s.pth'%(suffix)))
model.eval() #モデルを評価モードにする。
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Acc: {:.4f}'.format(acc))

print(type(data))