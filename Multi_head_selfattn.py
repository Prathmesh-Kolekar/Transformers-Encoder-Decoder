import torch.nn as nn 
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):

  def __init__(self):
    super().__init__()

  def scaled_dot_product_attention(self,q,k,v,mask):
    return F.scaled_dot_product_attention(q,k,v,mask)
    # q,k,v = q.to(device),k.to(device),v.to(device
    # print(q.shape,k.shape,v.shape)
    # return (F.softmax(q @ k.transpose(-2,-1) / k.size(2)**0.5) @ v).permute(0,2,1,3)

  def forward(self,qkv,mask=None):
    dim = qkv.size(-1)//3
    return self.scaled_dot_product_attention(qkv[:,:,:,:dim],qkv[:,:,:,dim:2*dim],qkv[:,:,:,2*dim:3*dim],mask=mask)