import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.rand(size=(self.in_features,self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.rand(size=(2*self.out_features,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        attention = self.getAttention(Wh)
        e = self.leakyrelu(torch.matmul(attention, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = torch.softmax(attention,dim=1)
        h_prime = torch.mm(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime



    def getAttention(self,Wh):
        N = Wh.size()[0]
        hi = Wh.repeat_interleave(N,dim=0)
        hj = Wh.repeat(N,1)
        attentionVal = torch.cat([hi,hj],dim=1)
        return attentionVal.view(N,N,-1)

class GAT(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_features, hidden, dropout, alpha, concat)
        self.layer2 = GATLayer(hidden, out_features, dropout, alpha, concat) 

    def forward(self, h, adj):
        h_prime = self.layer1(h, adj)
        result = self.layer2(h_prime, adj) 

        return F.log_softmax(result, dim=1)     

