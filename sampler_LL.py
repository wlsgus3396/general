import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy

class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model, LN,data, budget,task_model_previous,LN_previous,alpha):
        all_preds = []
        all_indices = []
        m=nn.Softmax(1)   
        for images, _, indices in data:
            preds=[]
            
            task_model.eval()
            LN.eval()

            if self.args.cuda:
                images = images.cuda()
                task_model=task_model.cuda()
                LN=LN.cuda()
            
            
            
            with torch.no_grad():    
                scores,features,_=task_model(images)
                scores_previous,features_previous,_=task_model_previous(images)
                p=LN(features)
                p_previous=LN_previous(features_previous)
                p = p.view(p.size(0))
                p_previous = p_previous.view(p_previous.size(0))
            for i in range(len(indices)):
                preds.append((1-alpha)*p[i]+alpha*p_previous[i])
                    
                    
                    
            preds = torch.tensor(preds,device='cpu')
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
    
        
        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.args.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
    