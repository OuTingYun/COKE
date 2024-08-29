import torch
import torch.nn as nn
from model.Decoder import *

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.W1 = nn.Linear(config.num_variables, config.n_samples)
        self.W2 = nn.Linear(config.n_samples, 1)
        self.relu = nn.ReLU()

    def forward(self, enc):
        mean_enc = torch.mean(enc, dim=-1)
        v = self.relu(self.W1(mean_enc))
        v = self.W2(v)
        return v.squeeze()
