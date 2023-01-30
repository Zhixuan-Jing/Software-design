import torch
import torch.nn as nn
import torch.nn.functional as F

# Example of a two-layered GCN
class GCN(nn.Module):
  # Parameters: 
  # in_feature: input feature dimension 
  # out_feature: class amount for classfier tasks
  # hidden_size: hidden layer size
  def __init__(self,in_feature,out_feature,hidden_size):
    super(GCN,self).__init__()
    # W1,W2 aer weight matrices
    self.W1 = nn.Parameter(torch.Tensor(in_feature,hidden_size))
    self.W2 = nn.Parameter(torch.Tensor(hidden_size,out_feature))
  
  # Parameters: 
  # adj: adjancency matrix of a graph, dimension n x n
  # nodes: node features, dimension n x in_features

  def forward(self,adj,nodes):
    # first node feature process
    nodes = torch.mm(nodes,self.W1)
    # convolution operation
    hid = torch.mm(adj,nodes)
    # second node feature process
    nodes = torch.mm(hid,self.W2)
    # convolution operation
    res = torch.mm(adj,nodes)

    return res

  def __repr__(self):
      return self.__class__.__name__ + ' ('             + str(self.input_dim) + ' -> '             + str(self.output_dim) + ')'
