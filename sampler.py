import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy
from seqsampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from torch.utils.data import DataLoader



class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model, data, budget,task_model_previous,alpha,data_unlabeled,subset,labeled_set,labeled_data_size):
        if self.args.execute=='RANDOM':

            all_indices = []

            for images, _, indices in data:
                all_indices.extend(indices)
            querry_indices=random.sample(range(len(all_indices)),budget)
            querry_pool_indices = np.asarray(all_indices)[querry_indices]
            
            return querry_pool_indices

        elif self.args.execute=='uncertainty' or self.args.execute=='F-uncertainty':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if self.args.cuda:
                    images = images.cuda()

            
                task_model.eval()
                task_model_previous.eval()

                if self.args.cuda:
                    task_model = task_model.cuda()
                    task_model_previous=task_model_previous.cuda()
                with torch.no_grad():    
                    p,_,_=task_model(images)
                    p_prev,_,_=task_model_previous(images)
                    p=m(p)
                    p_prev=m(p_prev)    
                    
                for i in range(len(indices)):
                    preds.append((1-alpha)*Categorical(probs = p[i,:]).entropy()+alpha*Categorical(probs = p_prev[i,:]).entropy())    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        
        
        elif self.args.execute=='MCdrop-entropy' or self.args.execute=='F-MCdrop-entropy':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if self.args.cuda:
                    images = images.cuda()

            
                task_model.eval()
                task_model_previous.eval()

                if self.args.cuda:
                    task_model = task_model.cuda()
                    task_model_previous = task_model_previous.cuda()
                with torch.no_grad():    
                    p,_,_=task_model(images)
                    p_prev,_,_=task_model_previous(images)
                    p=m(p)
                    p_prev=m(p_prev)
                    for _ in range(24):
                        p1,_,_=task_model(images)
                        p_prev1,_,_=task_model_previous(images)
                        p1=m(p1)
                        p_prev1=m(p_prev1)
                        
                        p+=p1
                        p_prev+=p_prev1    
                p=p/25
                p_prev=p_prev/25
                    
                for i in range(len(indices)):
                    preds.append((1-alpha)*Categorical(probs = p[i,:]).entropy()+alpha*Categorical(probs = p_prev[i,:]).entropy())    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        elif self.args.execute=='coreset' or self.args.execute=='F-coreset':
            
            unlabeled_loader = DataLoader(data_unlabeled, batch_size=self.args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)
            task_model.eval()
            features = torch.tensor([]).cuda()

            with torch.no_grad():
                for inputs, _, _ in unlabeled_loader:
                    
                    inputs = inputs.cuda()
                    _,_,features_batch = task_model(inputs)
                    features = torch.cat((features, features_batch), 0)
                
                feat = features.detach().cpu().numpy()
                new_av_idx = np.arange(len(subset),(len(subset) + labeled_data_size))
                sampling = kCenterGreedy(feat)  
                batch = sampling.select_batch_(new_av_idx, self.args.budget)
                other_idx = [x for x in range(len(subset)) if x not in batch]
            
            
            
            
            return other_idx + batch