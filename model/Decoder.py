import torch
import torch.nn as nn
import numpy as np
class Decoder(nn.Module):
    def __init__(self, batch_size, in_features,out_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.W = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self.flatten = nn.Flatten()

    def forward(self, enc):
        mask = torch.ones((self.batch_size, self.out_features)).to(enc.device)
        l_enc = self.flatten(enc)
        positions, log_softmaxs, errors = self.get_position(l_enc, mask)
        mask = mask - torch.nn.functional.one_hot(positions.squeeze(), num_classes=self.out_features)
        for i in range(self.out_features - 1):
            enc = enc * mask.unsqueeze(2)
            l_enc = self.flatten(enc)
            position, log_softmax, error = self.get_position(l_enc, mask)
            mask = mask - torch.nn.functional.one_hot(position.squeeze(), num_classes=self.out_features)
            log_softmaxs = log_softmaxs + log_softmax
            positions = torch.cat([position, positions], dim=1)
            errors+=error
        return positions, log_softmaxs, errors

    def get_position(self, l_enc, mask):
        p = self.W(l_enc)
        zero_vec = -9e15 * torch.ones_like(p)
        argsort_p = torch.argsort(p,descending=True).cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
        for i in range(int((1-mask_np).sum()/self.batch_size)+1):
            bigger_posi = argsort_p[:,i]
            error = 1*np.array([mask_np[idx,j]==0 for idx,j in enumerate(bigger_posi)])
        p = torch.where(mask == 0, zero_vec, p)
        log_p = torch.nn.functional.log_softmax(p, dim=1)
        position = torch.multinomial(log_p.exp(), num_samples=1)
        log_softmax = torch.sum(log_p * torch.nn.functional.one_hot(position.squeeze(), num_classes=self.out_features), dim=1)
        return position, log_softmax, error
