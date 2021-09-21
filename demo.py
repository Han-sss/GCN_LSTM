from functools import cmp_to_key
import pickle
from numpy.core.defchararray import array
from numpy.lib.function_base import piecewise
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pandas as pd
import datetime
from scipy.sparse import csr_matrix
import os.path
import embedding_help_functions as ehf
import random
import networkx as nx
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm

def func_create_sparse(A, N, TTT, T, start, end):
    assert (end-start) == T
    idx = (A._indices()[0] >= start) & (A._indices()[0] < end)        
    index = t.LongTensor(A._indices()[0:3,idx].size())
    values = t.DoubleTensor(A._values()[idx].size())    
    index[0:3] = A._indices()[0:3,idx]
    index[0] = index[0] - start
    values = A._values()[idx]
    sub = t.sparse.DoubleTensor(index, values , t.Size([T,N,N]))
    return sub.coalesce()

def create_matrix_M(T,no_diag):
    M = np.zeros((T,T))
    for i in range(no_diag):
        A = M[i:, :T-i]
        np.fill_diagonal(A, 1/(i+1))
        # print(M[19])
    # L = np.sum(M, axis=1)
    # M = M/L[:,None]
    M = t.tensor(M)
    return M

def func_MProduct_dense(C, M): # C is sparse input tensor, M is the inherent matrix used for generating M product
    T = C.shape[0] # step
    B = C.to_dense() 
    B = B.type(t.DoubleTensor)
    # print(B.size())
    X = t.matmul(M, B.reshape(T, -1)).reshape(B.size()) # figure.4 M-product 
    indices = t.nonzero(X).t()
    values = X[indices[0],indices[1],indices[2]] # modify this based on dimensionality
    Cm = t.sparse.DoubleTensor(indices, values, X.size())
    return Cm.to_dense()

# This function is used to construct augmented edge dataset for link prediction
def Augment_edges(edges, N, beta1, beta2, cutoff):
    edges_t = edges.transpose(1,0)
    edges_new = []
    
    # print(edges)
    # print(edges[0])


    for j in range(t.max(edges[0])+1):
        if j < cutoff:
            beta = beta1
        else:
            beta = beta2
        
        to_add = beta*t.sum(edges[0]==j)
        n_added = 0
        edges_subset = edges[1:3, edges[0]==j]
        print("the number of to be added edges is",to_add)

        while n_added < to_add:
            e = [random.randint(0,N-1), random.randint(0,N-1)]
            if t.max(t.sum(edges_subset.transpose(1,0) == t.tensor(e), 1)) < 2: # used to judge match or not
                edges_new.append([j, e[0], e[1]]) # add a edge of time 'j' 
                n_added += 1
        print(j)
        break
    
    edges_aug = t.cat((edges, t.tensor(edges_new).transpose(1,0)), 1)
    _, sort_id = edges_aug[0].sort()
    edges_aug = edges_aug[:, sort_id]
    edges_aug_t = edges_aug.transpose(1,0)
    
    print(edges.shape)
    print(edges_aug.shape)

    labels = t.cat((t.zeros(edges.shape[1], dtype=t.long), t.ones(edges_aug.shape[1]-edges.shape[1], dtype=t.long)), 0)
    labels = labels[sort_id] # added or not
    return edges_aug, labels

# Divide adjacency matrices and labels into training, validation and testing sets
def split_data(edges_aug, labels, S_train, S_val, S_test, same_block_size):
	# 	Training
	subs_train = edges_aug[0]<S_train
	edges_train = edges_aug[:, subs_train]
	target_train = labels[subs_train]
	e_train = edges_train[:, edges_train[0]!=0]
	e_train = e_train-t.cat((t.ones(1,e_train.shape[1]), t.zeros(2,e_train.shape[1])),0).long()

	#	Validation
	if same_block_size:
		subs_val = (edges_aug[0]>=S_val) & (edges_aug[0]<S_train+S_val)
	else:
		subs_val = (edges_aug[0]>=S_train) & (edges_aug[0]<S_train+S_val)
	edges_val = edges_aug[:, subs_val]
	if same_block_size:
		edges_val[0] -= S_val
	else:
		edges_val[0] -= S_train
	target_val = labels[subs_val]
	if same_block_size:
		K_val = t.sum(edges_val[0] - (S_train-S_val-1) > 0)
	e_val = edges_val[:, edges_val[0]!=0]
	e_val = e_val-t.cat((t.ones(1,e_val.shape[1]), t.zeros(2,e_val.shape[1])),0).long()

	#	Testing
	if same_block_size:
		subs_test = (edges_aug[0]>=S_test+S_val) 
	else:
		subs_test = (edges_aug[0]>=S_train+S_val)
	edges_test = edges_aug[:, subs_test]
	if same_block_size:
		edges_test[0] -= (S_test+S_val)
	else:
		edges_test[0] -= (S_train+S_val)
	target_test = labels[subs_test]
	if same_block_size:
		K_test = t.sum(edges_test[0] - (S_train-S_test-1) > 0)
	e_test = edges_test[:, edges_test[0]!=0]
	e_test = e_test-t.cat((t.ones(1,e_test.shape[1]), t.zeros(2,e_test.shape[1])),0).long()

	if same_block_size:
		return edges_train, target_train, e_train, edges_val, target_val, e_val, K_val, edges_test, target_test, e_test, K_test
	else:
		return edges_train, target_train, e_train, edges_val, target_val, e_val, edges_test, target_test, e_test



N = 200
T = 10
community_num = 2
node_change_num = 5
no_diag = 2
S_train = 7
S_val = 2
S_test = 1

beta1 = beta2 = 19
cutoff = 4

no_layers = 1
dataset = "OTC"
loss_type = "softmax" # "sigmoid" or "softmax"
eval_type = "MAP-MRR" # "MAP-MRR" or "F1"
no_epochs = 100

lr = 0.01
momentum = 0.9

dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(N,
                                                           community_num,
                                                           T,
                                                           1,  # comminity ID to perturb
                                                           node_change_num))

# graphs = [g[0] for g in dynamic_sbm_series]
graphs = list(map(lambda g:g[0], dynamic_sbm_series))
print( type( graphs[0] ) )
# print(list(graphs[0].nodes))
# print(list(graphs[0].edges))

A_sz = t.Size([N, N,1])

for tt in range(T):
    G = nx.adjacency_matrix(graphs[tt])
    ij = np.nonzero(G)
    vals = G[ij]
    ij = t.LongTensor(ij)
    # ij = t.cat((t.zeros((1,len(ij[0])),dtype=t.long), ij))
    vals  = t.DoubleTensor(vals).T
    tmp = t.sparse.DoubleTensor(ij, vals, A_sz)
    if tt==0:
        A = tmp
    else:
        A = t.cat((A,tmp),2)
    
X = A.to_dense()
X = X.transpose(2,0)        
indices = t.nonzero(X).t()
# print( X.size())

indices_tmp = t.nonzero(X)

values = X[indices[0],indices[1],indices[2]] # modify this based on dimensionality

# values_tmp = X[indices.t()]

A = t.sparse.DoubleTensor(indices, values, X.size())
# print(A.size())

C_train = func_create_sparse(A, N, T, S_train, 0, S_train)
C_val = func_create_sparse(A, N, T, S_train, S_val, S_val+S_train)
C_test = func_create_sparse(A, N, T, S_train, S_val+S_test, T)
# print(C_train.size())

M = create_matrix_M(S_train,no_diag) # B.2

Ct_train = func_MProduct_dense(C_train, M)#There is a bug in matlab Bitcoin Alpha
Ct_val = func_MProduct_dense(C_val, M)
Ct_test = func_MProduct_dense(C_test, M)

edges = A._indices()
# cholist = list(edges[0]==0)
# cholist = list(map(int, cholist))
# res = pd.value_counts(cholist)

# print(res)
# print(edges)
# print(edges.size())

# edges_subset = edges[1:3, edges[0]==0]
# print(edges_subset)
# print(edges_subset.size())

# e = [random.randint(0,N-1), random.randint(0,N-1)]
# print(e)
# sum_temp  = edges_subset.transpose(1,0) == t.tensor(e)
# sum_temp = list(t.sum(sum_temp,1))
# sum_temp = list(map(int,sum_temp))
# res = pd.value_counts(sum_temp)
# print(res)

# if t.max(t.sum(edges_subset.transpose(1,0) == t.tensor(e), 1)) < 2:
#     print("no match object")
# else:
#     print("AIM!")
# Create features for the nodes
X_train, X_val, X_test = ehf.create_node_features(A, S_train, S_val, S_test, same_block_size=True)


edges_aug, labels = Augment_edges(edges, N, beta1, beta2, cutoff)

edges_train, target_train, e_train, edges_val, target_val, e_val, K_val, edges_test, target_test, e_test, K_test \
= ehf.split_data(edges_aug, labels, S_train, S_val, S_test, same_block_size=True)

alpha_vec = [ .90]

for alpha in alpha_vec:
    class_weights = t.tensor([alpha, 1.0-alpha])
    save_res_fname = "results_OUR_layers" + str(no_layers) + "_w" + str(round(float(class_weights[0])*100)) + "_" + dataset + "_link_prediction"
    if loss_type == "softmax":
        n_output_feat = 2
    elif loss_type == "sigmoid":
        n_output_feat = 1

    if no_layers == 2: 
        gcn = ehf.EmbeddingGCN2(Ct_train[:-1], X_train[:-1], e_train, M[:-1, :-1], hidden_feat=[6,6,n_output_feat], condensed_W=True, use_Minv=False, nonlin2="selu")
    elif no_layers == 1:
        gcn = ehf.EmbeddingGCN(Ct_train[:-1], X_train[:-1], e_train, M[:-1, :-1], hidden_feat=[6,n_output_feat], condensed_W=True, use_Minv=False)
    
    # Train
    optimizer = t.optim.SGD(gcn.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Takes arguments (output, target)
    my_sig = nn.Sigmoid()
	
    if eval_type == "F1":
        ep_acc_loss = np.zeros((no_epochs,12)) # (precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test)
    elif eval_type == "MAP-MRR":
        ep_acc_loss = np.zeros((no_epochs,9)) # (MAP_train, MRR_train, loss_train, MAP_val, MRR_val, loss_val, MAP_test, MRR_test, loss_test)
			
    for ep in range(no_epochs):
        print("the current ep is",ep)
        optimizer.zero_grad()
        output_train = gcn()
        if loss_type == "sigmoid":
            p = my_sig(output_train)
            output_train = t.cat((p, 1-p), 1)
		
        loss_train = criterion(output_train, target_train[edges_train[0]!=0])
        loss_train.backward()
        optimizer.step()

        with t.no_grad():
            if ep % 50 == 0:
                # Compute stats for training data; no point in doing more often than this
                guess_train = t.argmax(output_train, dim=1)
                if eval_type == "F1":
                    precision_train, recall_train, f1_train = ehf.compute_f1(guess_train, target_train[edges_train[0]!=0])
                elif eval_type == "MAP-MRR":
                    MAP_train, MRR_train = ehf.compute_MAP_MRR(output_train, target_train[edges_train[0]!=0], edges_train[:, edges_train[0]!=0])

                # Compute stats for validation data
                output_val = gcn(Ct_val[:-1], X_val[:-1], e_val)
                if loss_type == "sigmoid":
                    p = my_sig(output_val)
                    output_val = t.cat((p, 1-p), 1)
                    
                guess_val = t.argmax(output_val, dim=1)
                if eval_type == "F1":
                    precision_val, recall_val, f1_val = ehf.compute_f1(guess_val[-K_val:], target_val[-K_val:])
                elif eval_type == "MAP-MRR":
                    MAP_val, MRR_val = ehf.compute_MAP_MRR(output_val[-K_val:], target_val[-K_val:], edges_val[:, -K_val:])
                    
                loss_val = criterion(output_val[-K_val:], target_val[-K_val:])

                # Compute stats for test data
                output_test = gcn(Ct_test[:-1], X_test[:-1], e_test)
                if loss_type == "sigmoid":
                    p = my_sig(output_test)
                    output_test = t.cat((p, 1-p), 1)
                    
                guess_test = t.argmax(output_test, dim=1)
                if eval_type == "F1":
                    precision_test, recall_test, f1_test = ehf.compute_f1(guess_test[-K_test:], target_test[-K_test:])
                
                elif eval_type == "MAP-MRR":
                    MAP_test, MRR_test = ehf.compute_MAP_MRR(output_test[-K_test:], target_test[-K_test:], edges_test[:, -K_test:])
                    
                loss_test = criterion(output_test[-K_test:], target_test[-K_test:])

                # Print
                if eval_type == "F1":
                    ehf.print_f1(precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test, alpha, tr, ep)
                elif eval_type == "MAP-MRR":
                    print("alpha/Tr/Ep %.2f/%d/%d. Train MAP/MRR %.16f/%.16f. Train loss %.16f." % (alpha, 1, ep, MAP_train, MRR_train, loss_train))
                    print("alpha/Tr/Ep %.2f/%d/%d. Val MAP/MRR %.16f/%.16f. Val loss %.16f." % (alpha, 1, ep, MAP_val, MRR_val, loss_val))
                    print("alpha/Tr/Ep %.2f/%d/%d. Test MAP/MRR %.16f/%.16f. Test loss %.16f.\n" % (alpha, 1, ep, MAP_test, MRR_test, loss_test))

				# Store values with results
                if eval_type == "F1":
                    ep_acc_loss[ep] = [precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test]
                elif eval_type == "MAP-MRR":
                    ep_acc_loss[ep] = [MAP_train, MRR_train, loss_train, MAP_val, MRR_val, loss_val, MAP_test, MRR_test, loss_test]
                    
        if eval_type == "F1":
            ehf.print_f1(precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test, is_final=True)
        elif eval_type == "MAP-MRR":
            print("FINAL: Train MAP/MRR %.16f/%.16f. Train loss %.16f." % (MAP_train, MRR_train, loss_train))
            print("FINAL: Val MAP/MRR %.16f/%.16f. Val loss %.16f." % (MAP_val, MRR_val, loss_val))
            print("FINAL: Test MAP/MRR %.16f/%.16f. Test loss %.16f.\n" % (MAP_test, MRR_test, loss_test))
        
        pickle.dump(ep_acc_loss, open(save_res_fname, "wb"))
        print("Results saved for single trial")
    