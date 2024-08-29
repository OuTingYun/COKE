import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT_Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT_Layer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.where(attention.isnan(),zero_vec,attention) #!for missing dataset

        attention = F.softmax(attention, dim=1)

        attention = attention.permute((0, 2, 1))
        attention = self.dropout(attention)
        Wh = torch.where(Wh.isnan(),torch.tensor([0]).to(Wh.device).float(),Wh) #!for missing dataset

        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + torch.transpose(Wh2, 1, 2)
        return self.leakyrelu(e)

class GAT_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nblocks,nvars):
        super(GAT_Encoder, self).__init__()
        self.dropout = dropout
        self.nblocks = nblocks
        self.nfeat = nfeat
        self.nhid = nhid
        self.alpha = alpha
        self.nheads = nheads
        self.nvars = nvars

        self.attentions = nn.ModuleList([
            nn.ModuleList([
                GAT_Layer(nfeat, int(nhid / nheads), dropout=dropout, alpha=alpha, concat=True)
                for _ in range(nheads)
            ])
            for _ in range(nblocks)
        ])
        self.init_normalization = nn.BatchNorm1d(self.nvars)
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(self.nvars) for _ in range(nblocks)])

    def forward(self, x, adj):
        '''
        x: (batch_size, nvars, nfeat)
        enc: (batch_size, nvars, nhid)
        '''
        enc = self.init_normalization(x)
        for i in range(self.nblocks):
            enc = torch.cat([att(enc, adj) for att in self.attentions[i]], dim=2)
            enc = self.normalizations[i](enc)
        return enc
