import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import numpy as np
import vgg
import sampler_LL
import copy
import random 






class Solver_2LL:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.sampler = sampler_LL.AdversarySampler(self.args)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    
    def train(self, querry_dataloader, val_dataloader, task_model,LN, unlabeled_dataloader,lr_input,momentum,iter):
       
        
        self.args.train_iterations = len(querry_dataloader)* self.args.train_epochs
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        
        if momentum=='True':
            optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input,weight_decay=5e-4, momentum=0.9)
            optim_LN = optim.SGD(LN.parameters(), lr=lr_input, weight_decay=5e-4, momentum=0.9)
        else:
            optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input)
            optim_LN = optim.SGD(LN.parameters(), lr=lr_input)
    
        task_model.train()
        LN.train()
        if self.args.cuda:
            task_model = task_model.cuda()
            LN=LN.cuda()
    
        
    
        for iter_count in range(self.args.train_iterations):
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            
            preds, features,_ =task_model(labeled_imgs)
            target_loss = self.ce_loss(preds, labels)
            if iter==-1:
                if iter_count>0.6*self.args.train_iterations:
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()
            else:
                if iter>0.6:
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()

            pred_loss=LN(features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = self.LossPredLoss(pred_loss, target_loss) 
            task_loss=m_backbone_loss +  m_module_loss
            
            
            
            optim_task_model.zero_grad()
            optim_LN.zero_grad()
            
            task_loss.backward()
            optim_task_model.step()
            optim_LN.step()
            
            
            
        
        
            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(m_backbone_loss.item()))
                print('Current task model loss: {:.4f}'.format(m_module_loss.item()))
                    
            
        
            
            
        
        
                
        task_model = task_model.cuda()
        final_accuracy=self.test(task_model)

        return final_accuracy,task_model,LN
        








    def sample_for_labeling(self, task_model , LN, unlabeled_dataloader,budget,task_model_previous,LN_previous,budgetratio):
        querry_indices = self.sampler.sample(task_model,LN,unlabeled_dataloader,budget,task_model_previous,LN_previous,budgetratio)

        return querry_indices
                





    def validate(self, task_model, loader):
        task_model.eval()
        loss=[]
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                preds,_,_ = task_model(imgs)
                loss.append(self.bcelogit_loss(np.squeeze(preds), labels.float()))
        
        
        return sum(loss)/len(loader)




    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,_,_ = task_model(imgs)
                preds= torch.round(torch.sigmoid(preds))
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    
    def cross_entropy(self,input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))
    
    
    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss