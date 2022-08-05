import copy
import torch
from torch import nn


def FedAvg2(w,n):
    W=[]
    for j in range(len(w)):
        w_avg = copy.deepcopy(w[j])
        for k in w_avg.keys():
            if k[len(k) - 3 :]=='ght' or k[len(k) - 3 :]=='ias': 
                w_avg[k]-=w[j][k]
                for i in range(0, len(w)):
                    w_avg[k] += w[i][k]*n[i]
                    
                w_avg[k] = torch.div(w_avg[k], sum(n))
        W.append(w_avg)
    return W