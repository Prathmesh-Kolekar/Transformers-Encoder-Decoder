import torch.nn as nn 
import torch.nn.functional as F

class FeedForward(nn.Module):
  def __init__(self,n_embedding):
    super().__init__()
    self.fcl1 = nn.Linear(n_embedding,2*n_embedding)
    self.fcl2 = nn.Linear(2*n_embedding,n_embedding)
    self.dropout = nn.Dropout(0.1)

  def forward(self,x):
    x = self.fcl1(x)
    x = F.relu(x)
    x = self.fcl2(x)
    x = F.relu(x)
    x = self.dropout(x)
    return x