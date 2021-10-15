from operator import mod
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

import GCN_model
import GCN_help_function as ghf

import dataset_apolloscape as APOL

epoch_num = 200
my_token = "t82"
suffix = "t1"
root_dir = '/home/mount/GCN-lstm/data/Apolloscape'
writer = SummaryWriter(comment='GCN_LSTM')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("The token of now is %s"%(my_token))

print("loading data ...")
APOL_dataset = APOL.trajectory_dataset(root_dir = root_dir, datatype = 'train')

APOL_dataloader = DataLoader(
        dataset=APOL_dataset,
        sampler=None,
        batch_size=1, # the graphs have a different number of edges, cannot stack the 'edge'
        num_workers=4,
        drop_last=True # ensure the lstm can work successfully
    )

model = GCN_model.Net(
        input_features=APOL_dataset.node_feature_num - 2, 
        output_features=APOL_dataset.class_num
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                            lr=0.0001,
                            betas=(0.9, 0.999),
                            eps = 1e-08,
                            weight_decay=5e-4)

print("loading finished!")

model.train()
acc_best = 0
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epoch_num):
    loss_sum = 0
    for input,result in APOL_dataloader:
        optimizer.zero_grad()

        node_matrixs = input['node_matrix']
        graphs = input['graph']
        
        node_matrix = node_matrixs[0][0]

        idx = torch.nonzero(node_matrix[:,1], as_tuple=True)[0]
        target = node_matrix[idx,2]-1

        node_matrix = node_matrix[:,(0,3,4,5,6,7,8,9)]
        graph = graphs[0][0]
        
        node_matrix = node_matrix.to(device)
        graph = graph.long().to(device)
        
        output = model(node_matrix,graph)
        output = output[idx]
        
        target = target.long().to(device)
        loss = criterion(output, target)
        loss_sum += loss
        
        loss.backward()
        optimizer.step()

    loss_sum = loss_sum.item()/len(APOL_dataloader)
    writer.add_scalar('Loss/train', loss_sum, epoch)
    print("the loss of epoch %d is: %f"%(epoch,loss_sum))

    if epoch % 10 == 0:
        with torch.no_grad():
            node_matrixs = APOL_dataset[0][1]['node_matrix']
            graphs = APOL_dataset[0][1]['graph']

            node_matrix = node_matrixs[0]
            
            idx = torch.nonzero(node_matrix[:,1],as_tuple=True)[0]
            target = node_matrix[idx,2]-1

            node_matrix = node_matrix[:,(0,3,4,5,6,7,8,9)]
            graph = graphs[0]

            node_matrix = node_matrix.to(device)
            graph = graph.long().to(device)
            
            _,pred = model(node_matrix,graph).max(dim=1)
            pred = pred[idx]
            target = target.long().to(device)
            
            correct = float(pred.eq(target).sum().item())
            acc = correct / len(idx)

            writer.add_scalar('Acc/train', acc, epoch)
                
            if acc > acc_best:
                acc_best = acc
                torch.save(model.state_dict(),'GCN_model_params_%s.pth'%(suffix))
                print('Acc of epoch {}: {:.4f}'.format(epoch,acc))
                



model.load_state_dict(torch.load('GCN_model_params_%s.pth'%(suffix)))

for name, param in model.named_parameters():
    if param.requires_grad:
        if (name == 'L_temp.weight') | (name == 'L_temp.bias'):
            print(name)
            print(param.data)

model.eval() #モデルを評価モードにする。

node_matrixs = APOL_dataset[100][1]['node_matrix']
graphs = APOL_dataset[100][1]['graph']

node_matrix = node_matrixs[0]
            
idx = torch.nonzero(node_matrix[:,1],as_tuple=True)[0]
target = node_matrix[idx,2]-1

node_matrix = node_matrix[:,(0,3,4,5,6,7,8,9)]
graph = graphs[0]

node_matrix = node_matrix.to(device)
graph = graph.long().to(device)

_,pred = model(node_matrix,graph).max(dim=1)
pred = pred[idx]
target = target.long().to(device)

correct = float(pred.eq(target).sum().item())
acc = correct / len(idx)

print('The final Acc of test is: {:.4f}'.format(acc))
writer.add_scalar('Acc/test', acc)
writer.close()