import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy

class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model, vae,discriminator,data, budget,task_model_previous,vae_previous,discriminator_previous,alpha):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if self.args.cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

                _, _, mu_previous, _ = vae_previous(images)
                preds_previous = discriminator_previous(mu_previous)

            preds = preds.cpu().data
            preds_previous = preds_previous.cpu().data
            all_preds.extend((1-alpha)*preds+alpha*preds_previous)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.args.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]
        return querry_pool_indices