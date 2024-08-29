import torch
import torch.nn as nn
from model.GAT_Encoder import *
from model.Decoder import *

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.use_x_miss = config.use_x_miss
        self.use_x_full = config.use_x_full
        self.encoder = GAT_Encoder(config.n_samples, config.n_samples, config.dropout, config.alpha, config.nheads, config.nblocks, config.num_variables)
        self.decoder = Decoder(config.batch_size, config.num_variables*config.n_samples, config.num_variables)
        self.theta_full = nn.Parameter(torch.zeros(1)+config.theta_full)
        self.theta_miss = nn.Parameter(torch.zeros(1)-config.theta_miss)
    def forward(self, X:tuple[torch.Tensor,torch.Tensor], adj:torch.Tensor,i=0):
        '''
        X_full: torch.tensor (bs, num_variables, n_samples)
        X_miss: torch.tensor (num_recipes, bs, num_variables, n_samples)
        adj: torch.tensor (num_variables, num_variables)
        '''
        
        X_full, X_miss = X[0], X[1]
        #* x_full
        if self.use_x_full and not self.use_x_miss: 
            if i>0 and i%200==0:print("\nonly use x_full")
            enc = self.encoder(X_full, adj)

        #* x_miss + x_full
        elif self.use_x_full and self.use_x_miss: 
            if i>0 and i%200==0:
                print("\nuse x_miss and x_full")
                print(self.theta_full,self.theta_miss)
            enc = self.encoder(X_full, adj)
            enc_miss_stack = []
            for rep in range(X_miss.size(0)):
                enc_miss = self.encoder(X_miss[rep], adj)
                enc_miss_stack.append(enc_miss)
            enc_miss_stack = torch.stack(enc_miss_stack)
            enc_miss_mean = torch.mean(enc_miss_stack, dim=0)
            enc = self.theta_full*enc + self.theta_miss*enc_miss_mean

        #* x_miss
        elif not self.use_x_full and self.use_x_miss: 
            if i>0 and i%200==0:print("\nonly use x_miss")
            enc_miss_stack = []
            for rep in range(X_miss.size(0)):
                enc_miss = self.encoder(X_miss[rep], adj)
                enc_miss_stack.append(enc_miss)
            enc_miss_stack = torch.stack(enc_miss_stack)
            enc_miss_mean = torch.mean(enc_miss_stack, dim=0)
            enc = enc_miss_mean

        positions, log_softmaxs, errors = self.decoder(enc)
        # if i%200==0:print(errors)
        return enc, positions, log_softmaxs, errors
