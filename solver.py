import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import numpy as np
import vgg
import sampler
import copy
import random 






class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sampler = sampler.AdversarySampler(self.args)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    
    def train(self, querry_dataloader, val_dataloader, task_model, unlabeled_dataloader,lr_input,momentum):
       
        
        self.args.train_iterations = len(querry_dataloader)* self.args.train_epochs
        labeled_data = self.read_data(querry_dataloader)
        
        
        if momentum=='True':
            optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input,weight_decay=5e-4, momentum=0.9)
        else:
            optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input)
    
        task_model.train()
        if self.args.cuda:
            task_model = task_model.cuda()
        
    
        for iter_count in range(self.args.train_iterations):
            

            labeled_imgs, labels = next(labeled_data)
            

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()


            preds,_,_=task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()
            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                
                
        
        
        task_model = task_model.cuda()
        final_accuracy=self.test(task_model)

        return final_accuracy,task_model
        




    def sample_for_labeling(self, task_model, unlabeled_dataloader,budget,task_model_previous,budgetratio,data_unlabeled,subset,labeled_set,labeled_data_size):
        querry_indices = self.sampler.sample(task_model, unlabeled_dataloader,budget,task_model_previous,budgetratio,data_unlabeled,subset,labeled_set,labeled_data_size)
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
                loss.append(self.ce_loss(preds, labels))
        
        
        return sum(loss)/len(loader)




    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,_,_ = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    
    
    def dis(self,p,p1,p2):
        dis=0
        m=nn.Softmax(1) 
        dis+=sum(sum(abs(m(p)-m(p1))))
        dis+=sum(sum(abs(m(p)-m(p2))))
        dis+=sum(sum(abs(m(p1)-m(p2))))
                
        dis=dis/self.args.num_classes/len(p[:,0])
        
        return dis
    
    
    def DLcal(self,f,f1,f2,loader):
        dis=0
        task_model1=copy.deepcopy(f)
        task_model2=copy.deepcopy(f)
        task_model1.linear=f1
        task_model2.linear=f2
        m=nn.Softmax(1) 
        f.eval()
        task_model1.eval()
        task_model2.eval()
        if self.args.cuda:
            f = f.cuda()
            task_model1 = task_model1.cuda()
            task_model2 = task_model2.cuda()
                    
        for imgs, _, _ in loader:
            imgs=imgs.cuda()
            with torch.no_grad():
                p,_,_=f(imgs)
                p1,_,_=f(task_model1(imgs))
                p2,_,_=f(task_model2(imgs))
                p=m(p)
                p1=m(p1)
                p2=m(p2)
            dis+=sum(sum(abs(p-p1)))
            dis+=sum(sum(abs(p-p2)))
            dis+=sum(sum(abs(p1-p2)))
            
       
        return dis
    
    def cross_entropy(self,input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))