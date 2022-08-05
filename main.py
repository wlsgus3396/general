import torch
from torchvision import datasets, transforms, models
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import torch.nn as nn

import numpy as np
import argparse
import random
import os
import LL
from custom_datasets import *
import model
import resnet
import resnet_drop
import resnet_bn
import resnet_bn_drop
from solver import Solver
from solver_dis import Solver_dis
from solver_LL import Solver_LL
from solver_2LL import Solver_2LL
from solver_vaal import Solver_vaal
from utils import *
import arguments
import copy
from FedAVG import FedAvg
from FedAVG2 import FedAvg2




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
def mnist_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.286), (0.353))
       ])
def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)

        train_dataset = CIFAR10(args.data_path)
        untrain_dataset=plain_CIFAR10(args.data_path)
        args.num_images = 50000
        args.num_val = 5000

        args.num_classes = 10
        
    elif args.dataset == 'mnist':
        test_dataloader = data.DataLoader(
                datasets.FashionMNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)
        untrain_dataset=plain_MNIST(args.data_path)
        train_dataset = MNIST(args.data_path)
        
        
        args.num_images = 50000
        args.num_val = 5000

        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)

        train_dataset = CIFAR100(args.data_path)
        untrain_dataset=plain_CIFAR100(args.data_path)
        args.num_images = 50000
        args.num_val = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError


    GPU_NUM = args.gpu
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    random_seed=1000*args.K
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





    all_indices = set(np.arange(args.num_images)) 
    labeled_indices=[]
    unlabeled_indices=[]
    accuracy=[]
    Solo_accuracy=[]
    accuracies = []
    args.cuda=torch.cuda.is_available()





    all_indices1=copy.deepcopy(all_indices)
    if args.iid=='True':
        for k in range(args.num_clients):
            unlabeled_indices.append(list(random.sample(list(all_indices1), args.unlabeledbudget)))
            all_indices1=np.setdiff1d(list(all_indices1), unlabeled_indices[k])
            labeled_indices.append(list(random.sample(list(unlabeled_indices[k]), args.initial_budget)))
            unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), labeled_indices[k]))
            accuracies.append(0)


    else:
        if args.dataset=='cifar10':
            unlabeled_indices=noniid(datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=True),args.num_clients)
        elif args.dataset=='mnist':     
            unlabeled_indices=noniid(datasets.MNIST(args.data_path, download=True, transform=mnist_transformer(), train=True),args.num_clients)
        for k in range(args.num_clients):
            all_indices1=np.setdiff1d(list(all_indices1), unlabeled_indices[k])
            labeled_indices.append(list(random.sample(list(unlabeled_indices[k]), args.initial_budget))) 
            unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), labeled_indices[k]))
            accuracies.append(0)
    val_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    








    for iiter in range(args.global_iteration1): 
        Solo_accuracy.append(0)
        unlabeled_dataloader=[]
        unlabeled_selected_dataloader=[]
        querry_dataloader=[] 
        val_dataloader=[]   
        n_avg=[]
        for k in range(args.num_clients):
                unlabeled_indices[k].sort()
                labeled_indices[k].sort()
                unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices[k])
                unlabeled_dataloader.append(data.DataLoader(untrain_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker))
                
                sampler = data.sampler.SubsetRandomSampler(labeled_indices[k])
                unlabeled_selected_dataloader.append(data.DataLoader(untrain_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker))  
                querry_dataloader.append(data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker))
                n_avg.append(len(labeled_indices[k]))
                val_dataloader.append(data.DataLoader(untrain_dataset, sampler=sampler,batch_size=args.batch_size, drop_last=False))
        



        





        random_seed=100+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        task_model=[]
        solver=[]
        if args.bn=='True':
            Itask_model=resnet_bn.resnet18()
        else:
            Itask_model=resnet.resnet18()
        

        
        for k in range(args.num_clients):
            solver.append(Solver(args, test_dataloader))
            if args.bn=='True':
                task_model.append(resnet_bn.resnet18())
            else:
                task_model.append(resnet.resnet18())


        num_ftrs = task_model[0].linear.in_features



        
        
        lr=args.lr
        random_seed=200+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        AVG_accuracy=0
       
        
        
        
        for iter in range(args.global_iteration2): 
            
            if iter!=0:
                lr*=args.lr_decay
            w_task_model=[]
           



            for k in range(args.num_clients):
                     
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))


                _,task_model[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],unlabeled_dataloader[k],lr,args.momentum)
                w_task_model.append(task_model[k].state_dict())
                





            if args.bn=='True':
                WW=FedAvg2(w_task_model,n_avg)
            else:
                WW=FedAvg(w_task_model,n_avg)
            

            if iter!=args.global_iteration2-1:
                for k in range(args.num_clients):
                    if args.bn=='True':
                        task_model[k].load_state_dict(copy.deepcopy(WW[k]))
                    else:
                        task_model[k].load_state_dict(copy.deepcopy(WW))

        
        for k in range(args.num_clients):
            task_model[k] = task_model[k].cuda()    
            AVG_accuracy+= solver[k].test(task_model[k])        
        
        AVG_accuracy=AVG_accuracy/args.num_clients
        accuracy.append(AVG_accuracy)        
    

        

        
     
###########################################################################################################################################################################################################################################
        random_seed=110+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

        solver=[]
        task_model=[]
        if args.execute=='F-dis': 
            FC1=[]
            FC2=[]
        elif args.execute=='F-LL': 
            LN=[]
        elif args.execute=='F-2LL': 
            LN=[]
        elif args.execute=='F-vaal':
            vae=[]
            discriminator=[]   

        if args.bn=='True':
            if args.execute=='F-MCdrop-entropy':    #######MCdrop-entropy를 뺀 이유: 여기서 MCdrop-entropy의 F accuracy를 얻고 나중에 task_model_previous를 얻어서 일괄성 있게 하려고 
                Itask_model=resnet_bn_drop.resnet18()
            else:
                Itask_model=resnet_bn.resnet18()
        else:
            if args.execute=='F-MCdrop-entropy':     
                Itask_model=resnet_drop.resnet18()
            else:
                Itask_model=resnet.resnet18()
        


        if args.execute=='F-dis':
            num_ftrs = Itask_model.linear.in_features
            FC2linear=nn.Linear(num_ftrs,args.num_classes) 
        elif args.execute=='F-LL': 
            ILN=LL.LossNet().cuda()
        elif args.execute=='F-2LL': 
            ILN=LL.LossNet().cuda()
        elif args.execute=='F-vaal':
            Ivae=model.VAE(args.latent_dim)
            Idiscriminator=model.Discriminator(args.latent_dim)
        
        
        for k in range(args.num_clients):
            if args.execute=='F-dis':
                solver.append(Solver_dis(args, test_dataloader))
            elif args.execute=='F-LL': 
                solver.append(Solver_LL(args, test_dataloader))
            elif args.execute=='F-2LL': 
                solver.append(Solver_2LL(args, test_dataloader))
            elif args.execute=='F-vaal':
                solver.append(Solver_vaal(args, test_dataloader))
            else:
                solver.append(Solver(args, test_dataloader))



            if args.bn=='True':
                if args.execute=='F-MCdrop-entropy':
                    task_model.append(resnet_bn_drop.resnet18())
                else:
                    task_model.append(resnet_bn.resnet18())
            else:
                if args.execute=='F-MCdrop-entropy':     
                    task_model.append(resnet_drop.resnet18())
                else:
                    task_model.append(resnet.resnet18())

            if args.execute=='F-dis':
                FC1.append(nn.Linear(num_ftrs,args.num_classes))
                FC1[k].load_state_dict(Itask_model.linear.state_dict())
                FC2.append(nn.Linear(num_ftrs,args.num_classes))
                FC2[k].load_state_dict(FC2linear.state_dict())
            elif args.execute=='F-LL': 
                LN.append(LL.LossNet().cuda())
                LN[k].load_state_dict(ILN.state_dict())
            elif args.execute=='F-2LL': 
                LN.append(LL.LossNet().cuda())
                LN[k].load_state_dict(ILN.state_dict())
            elif args.execute=='F-vaal':
                vae.append(model.VAE(args.latent_dim))
                vae[k].load_state_dict(Ivae.state_dict())
                discriminator.append(model.Discriminator(args.latent_dim))
                discriminator[k].load_state_dict(Idiscriminator.state_dict())





        
        
        lr=args.lr
        random_seed=210+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        
        
        
        
        for iter in range(args.global_iteration2): 
            if args.execute=='Random' or args.execute=='uncertainty' or args.execute=='dis' or args.execute=='coreset' or args.execute=='LL' or args.execute=='2LL' or args.execute=='MCdrop-entropy' or args.execute=='vaal':
                break

            if iter!=0:
                lr*=args.lr_decay
            w_task_model=[]
            if args.execute=='F-dis':
                w_FC1=[]
                w_FC2=[]
            elif args.execute=='F-LL': 
                w_LN=[]
            elif args.execute=='F-2LL': 
                w_LN=[]
            elif args.execute=='F-vaal':
                w_vae=[]
                w_discriminator=[]
           



            for k in range(args.num_clients):
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))

                
                

                if args.execute=='F-dis':
                    _,task_model[k],FC1[k],FC2[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],FC1[k],FC2[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='F-LL': 
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='F-2LL':
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum,iter/args.global_iteration2)
                elif args.execute=='F-vaal':
                    _,task_model[k],vae[k],discriminator[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator,unlabeled_dataloader[k],lr,args.momentum)
                else:
                    _,task_model[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],unlabeled_dataloader[k],lr,args.momentum)
                




                w_task_model.append(task_model[k].state_dict())
                if args.execute=='F-dis':
                    w_FC1.append(FC1[k].state_dict())
                    w_FC2.append(FC2[k].state_dict())
                elif args.execute=='F-LL': 
                    w_LN.append(LN[k].state_dict())
                elif args.execute=='F-2LL':
                    w_LN.append(LN[k].state_dict())
                elif args.execute=='F-vaal':
                    w_vae.append(vae[k].state_dict())
                    w_discriminator.append(discriminator[k].state_dict())






            if args.bn=='True':
                WW=FedAvg2(w_task_model,n_avg)
            else:
                WW=FedAvg(w_task_model,n_avg)
            
            if args.execute=='F-dis':
                global_FC1=copy.deepcopy(FC1[0])
                global_FC1.load_state_dict(FedAvg(w_FC1,n_avg))
                global_FC2=copy.deepcopy(FC2[0])
                global_FC2.load_state_dict(FedAvg(w_FC2,n_avg))
            elif args.execute=='F-LL': 
                global_LN=copy.deepcopy(LN[0])
                global_LN.load_state_dict(FedAvg(w_LN,n_avg))
            elif args.execute=='F-2LL':
                global_LN=copy.deepcopy(LN[0])
                global_LN.load_state_dict(FedAvg(w_LN,n_avg))
            elif args.execute=='F-vaal':
                global_vae=copy.deepcopy(vae[0])
                global_vae.load_state_dict(FedAvg(w_vae,n_avg))
                global_discriminator=copy.deepcopy(discriminator[0])
                global_discriminator.load_state_dict(FedAvg(w_discriminator,n_avg))

            if iter!=args.global_iteration2-1:
                for k in range(args.num_clients):
                    if args.bn=='True':
                        task_model[k].load_state_dict(copy.deepcopy(WW[k]))
                    else:
                        task_model[k].load_state_dict(copy.deepcopy(WW))
                    
                    
                    
                    if args.execute=='F-dis':
                        FC1[k]=global_FC1
                        FC2[k]=global_FC2
                    elif args.execute=='F-LL': 
                        LN[k]=global_LN
                    elif args.execute=='F-2LL':
                        LN[k]=global_LN
                    elif args.execute=='F-vaal':
                        vae[k]=global_vae
                        discriminator[k]=global_discriminator

        

        task_model_previous=[]
        for k in range(args.num_clients):
            task_model_previous.append(task_model[k])
        
        if args.execute=='F-dis':
            FC1_previous=[]
            FC2_previous=[]
            for k in range(args.num_clients):
                FC1_previous.append(FC1[k])
                FC2_previous.append(FC2[k])
        elif args.execute=='F-LL': 
            LN_previous=[]
            for k in range(args.num_clients):
                LN_previous.append(LN[k])
        elif args.execute=='F-2LL': 
            LN_previous=[]
            for k in range(args.num_clients):
                LN_previous.append(LN[k])
        elif args.execute=='F-vaal':
            vae_previous=[]
            discriminator_previous=[]
            for k in range(args.num_clients):
                vae_previous.append(vae[k])
                discriminator_previous.append(discriminator[k])
        
        if args.execute=='F-dis': 
            DL_dis_previous=0
            Num_dis_previous=0
            DL_item_previous=[]
            for k in range(args.num_clients):
                DL_dis_previous+=solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])
                Num_dis_previous+=len(labeled_indices[k])
                DL_item_previous.append(solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])/len(labeled_indices[k])/args.num_classes)
        
            DL_dis_previous=DL_dis_previous/Num_dis_previous/args.num_classes



#################################################################################################################################################################################################################################
        
        
        
        
        
        
        
        
        
        
        random_seed=300+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        solver=[]
        task_model=[]
        
       
        for k in range(args.num_clients):
            solver.append(Solver(args, test_dataloader))
            if args.bn=='True':
                task_model.append(resnet_bn.resnet18())
            else:
                task_model.append(resnet.resnet18())
            task_model[k].linear=nn.Linear(num_ftrs,args.num_classes)
                

        
        lr=args.lr_solo
        random_seed=400+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for iter in range(1):   
            for k in range(args.num_clients):   
                
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))
                print('Solo learning')
                Soloaccuracy, task_model[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],unlabeled_dataloader[k],lr,args.momentum)
                Solo_accuracy[iiter]+=Soloaccuracy

        Solo_accuracy[iiter]=Solo_accuracy[iiter]/args.num_clients


#######################################################################################################################################################################################################################################

        random_seed=310+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        solver=[]
        task_model=[]
        if args.execute=='dis' or args.execute=='F-dis': 
            FC1=[]
            FC2=[]
        elif args.execute=='LL' or args.execute=='F-LL': 
            LN=[]
        elif args.execute=='2LL' or args.execute=='F-2LL': 
            LN=[]
        elif args.execute=='vaal' or args.execute=='F-vaal':
            vae=[]
            discriminator=[]     

        
        
        for k in range(args.num_clients):
            if args.execute=='dis' or args.execute=='F-dis':
                solver.append(Solver_dis(args, test_dataloader))
            elif args.execute=='LL' or args.execute=='F-LL': 
                solver.append(Solver_LL(args, test_dataloader))
            elif args.execute=='2LL' or args.execute=='F-2LL': 
                solver.append(Solver_2LL(args, test_dataloader))
            elif args.execute=='vaal' or args.execute=='F-vaal':
                solver.append(Solver_vaal(args, test_dataloader))
            else:
                solver.append(Solver(args, test_dataloader))



            if args.bn=='True':
                if args.execute=='MCdrop-entropy' or args.execute=='F-MCdrop-entropy':
                    task_model.append(resnet_bn_drop.resnet18())
                else:
                    task_model.append(resnet_bn.resnet18())
            else:
                if args.execute=='MCdrop-entropy' or args.execute=='F-MCdrop-entropy':     
                    task_model.append(resnet_drop.resnet18())
                else:
                    task_model.append(resnet.resnet18())

            if args.execute=='dis' or args.execute=='F-dis':
                FC1.append(nn.Linear(num_ftrs,args.num_classes))
                FC2.append(nn.Linear(num_ftrs,args.num_classes))
                FC1[k].load_state_dict(task_model[k].linear.state_dict())
            elif args.execute=='LL' or args.execute=='F-LL': 
                LN.append(LL.LossNet().cuda())
            elif args.execute=='2LL' or args.execute=='F-2LL': 
                LN.append(LL.LossNet().cuda())
            elif args.execute=='vaal' or args.execute=='F-vaal':
                vae.append(model.VAE(args.latent_dim))
                discriminator.append(model.Discriminator(args.latent_dim))




        
        
        lr=args.lr_solo
        random_seed=410+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for iter in range(1): 
            

            for k in range(args.num_clients):
                     
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))
                print('Solo learning')  
                

                if args.execute=='dis':
                    _,task_model[k],FC1[k],FC2[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],FC1[k],FC2[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='F-dis':
                    _,task_model[k],FC1[k],FC2[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],FC1[k],FC2[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='LL': 
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='F-LL': 
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='2LL':
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum,-1)
                elif args.execute=='F-2LL':
                    _,task_model[k],LN[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],LN[k],unlabeled_dataloader[k],lr,args.momentum,-1)
                elif args.execute=='vaal':
                    _,task_model[k],vae[k],discriminator[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator[k],unlabeled_dataloader[k],lr,args.momentum)
                elif args.execute=='F-vaal':
                    _,task_model[k],vae[k],discriminator[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator[k],unlabeled_dataloader[k],lr,args.momentum)
                else:
                    _,task_model[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],unlabeled_dataloader[k],lr,args.momentum)
                

        
  
###############################################################################################################################################################################################################################
        
        
        if args.execute=='dis' or args.execute=='F-dis': 
            DL_dis=0
            Num_dis=0
            DL_item=[]
            for k in range(args.num_clients):
                DL_dis+=solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])
                Num_dis+=len(labeled_indices[k])
                DL_item.append(solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])/len(labeled_indices[k])/args.num_classes)
        
            DL_dis=DL_dis/Num_dis/args.num_classes


        if args.execute=='coreset':
            for k in range(args.num_clients):
                arg = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model[k],0,untrain_dataset,unlabeled_indices[k],labeled_indices[k],len(labeled_indices[k]))
                sampled_indices=list(torch.tensor(unlabeled_indices[k])[arg][-args.budget:].numpy())
                labeled_indices[k]= list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))

        elif args.execute=='F-coreset':
            for k in range(args.num_clients):
                arg = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model[k],0,untrain_dataset,unlabeled_indices[k],labeled_indices[k],len(labeled_indices[k]))
                sampled_indices=list(torch.tensor(unlabeled_indices[k])[arg][-args.budget:].numpy())
                labeled_indices[k]= list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
        
        elif args.execute=='dis':
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],FC1[k], FC2[k], DL_dis ,DL_item[k] ,unlabeled_dataloader[k],int(args.budget),task_model[k],FC1[k], FC2[k],DL_dis ,DL_item[k],0)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))

        
        elif args.execute=='F-dis': 
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],FC1[k], FC2[k], DL_dis ,DL_item[k] ,unlabeled_dataloader[k],int(args.budget),task_model_previous[k],FC1_previous[k], FC2_previous[k],DL_dis_previous ,DL_item_previous[k],args.budgetratio)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))

        elif args.execute=='LL' or args.execute=='2LL':  
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],LN[k] ,unlabeled_dataloader[k],int(args.budget),task_model[k],LN[k],0)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
    
        elif args.execute=='F-LL' or args.execute=='F-2LL':  
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],LN[k] ,unlabeled_dataloader[k],int(args.budget),task_model_previous[k],LN_previous[k],args.budgetratio)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
    
        
        
        elif args.execute=='vaal': 
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],vae[k] , discriminator[k], unlabeled_dataloader[k],int(args.budget),task_model[k],vae[k],discriminator[k],0)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
    

        elif args.execute=='F-vaal': 
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],vae[k] , discriminator[k], unlabeled_dataloader[k],int(args.budget),task_model_previous[k],vae_previous[k],discriminator_previous[k],args.budgetratio)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
    
        elif args.execute=='RANDOM':
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model[k],0)
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
        elif args.execute=='uncertainty':
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model[k],0,untrain_dataset,unlabeled_indices[k],labeled_indices[k],len(labeled_indices[k]))
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
        elif args.execute=='MCdrop-entropy':
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model[k],0,untrain_dataset,unlabeled_indices[k],labeled_indices[k],len(labeled_indices[k]))
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
        else:
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k], unlabeled_dataloader[k],int(args.budget),task_model_previous[k],args.budgetratio,untrain_dataset,unlabeled_indices[k],labeled_indices[k],len(labeled_indices[k]))
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
    
        print('Final Fed accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, AVG_accuracy))
        print('Final Solo accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, Solo_accuracy[iiter]))
        


    #torch.save(accuracy, os.path.join(args.out_path, args.log_name))
    accuracy = np.array(accuracy)
    A ="\n".join(map(str, accuracy))
    f = open('./results/{}_budgetratio_{}_iid_{}_bn_{}_momentum_{}_Fedacc_{}.csv'.format(args.execute,args.budgetratio,args.iid,args.bn,args.momentum,args.K),'w')
    f.write(A)
    f.close()

    Solo_accuracy=np.array(Solo_accuracy)
    A ="\n".join(map(str, Solo_accuracy))
    f = open('./results/{}_budgetratio_{}_iid_{}_bn_{}_momentum_{}_Soloacc_{}.csv'.format(args.execute,args.budgetratio,args.iid,args.bn,args.momentum,args.K),'w')
    f.write(A)
    f.close()

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

