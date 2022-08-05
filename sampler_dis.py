import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy

class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model,FC1,FC2,DL_F,DL_I,data, budget,task_model_previous,FC1_previous,FC2_previous,DL_F_previous,DL_I_previous,alpha):
        
        all_preds = []
        all_indices = []
        m=nn.Softmax(1)   
        for images, _, indices in data:
            preds=[]
            if self.args.cuda:
                images = images.cuda()

            
            task_model1=copy.deepcopy(task_model)
            task_model2=copy.deepcopy(task_model)
            task_model1.linear=FC1
            task_model2.linear=FC2
            task_model.eval()
            task_model1.eval()
            task_model2.eval()    
            if self.args.cuda:
                task_model = task_model.cuda()
                task_model1 = task_model1.cuda()
                task_model2 = task_model2.cuda()
            with torch.no_grad():    
                p,_,_=task_model(images)
                p1,_,_=task_model1(images)
                p2,_,_=task_model2(images)
                p=m(p)
                p1=m(p1)
                p2=m(p2)

            task_model1_previous=copy.deepcopy(task_model_previous)
            task_model2_previous=copy.deepcopy(task_model_previous)
            task_model1_previous.linear=FC1_previous
            task_model2_previous.linear=FC2_previous
            task_model_previous.eval()
            task_model1_previous.eval()
            task_model2_previous.eval()    
            if self.args.cuda:
                task_model_previous = task_model_previous.cuda()
                task_model1_previous = task_model1_previous.cuda()
                task_model2_previous = task_model2_previous.cuda()
            with torch.no_grad():    
                p_previous,_,_=task_model_previous(images)
                p1_previous,_,_=task_model1_previous(images)
                p2_previous,_,_=task_model2_previous(images)
                p_previous=m(p_previous)
                p1_previous=m(p1_previous)
                p2_previous=m(p2_previous)

            for i in range(len(indices)):
                if self.args.execute=='F-dis':
                    x=(1-alpha)*abs(sum(abs(p[i,:]-p1[i,:]))/len(p[0,:])+sum(abs(p[i,:]-p2[i,:]))/len(p[0,:]) + sum(abs(p1[i,:]-p2[i,:]))/len(p[0,:])-DL_F)
                    x+=alpha*abs(sum(abs(p_previous[i,:]-p1_previous[i,:]))/len(p_previous[0,:])+sum(abs(p_previous[i,:]-p2_previous[i,:]))/len(p_previous[0,:]) + sum(abs(p1_previous[i,:]-p2_previous[i,:]))/len(p_previous[0,:])-DL_F_previous)
                    preds.append(x)
                else:
                    x=(1-alpha)*abs(sum(abs(p[i,:]-p1[i,:]))/len(p[0,:])+sum(abs(p[i,:]-p2[i,:]))/len(p[0,:]) + sum(abs(p1[i,:]-p2[i,:]))/len(p[0,:])-DL_I)
                    x+=alpha*abs(sum(abs(p_previous[i,:]-p1_previous[i,:]))/len(p_previous[0,:])+sum(abs(p_previous[i,:]-p2_previous[i,:]))/len(p_previous[0,:]) + sum(abs(p1_previous[i,:]-p2_previous[i,:]))/len(p_previous[0,:])-DL_I_previous)
                    preds.append(x)
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
    