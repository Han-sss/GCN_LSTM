from locale import ABDAY_4
import torch

def is_symmetric(A_sparse):
    A_dense = A_sparse.to_dense()
    res = (A_dense.transpose(0,1) == A_dense).all()
    return res