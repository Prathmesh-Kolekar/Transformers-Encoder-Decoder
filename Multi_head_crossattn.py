import torch.nn as nn 
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):

  def __init__(self,n_embedding):
    super().__init__()
    self.n_embedding = n_embedding

  def forward(self,q,kv,mask):
    dim = kv.size(-1)//2
    print(q.shape,kv.shape)
    return F.scaled_dot_product_attention(q,kv[:,:,:,:dim],kv[:,:,:,dim:],mask)