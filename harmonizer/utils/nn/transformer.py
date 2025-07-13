import torch.nn as nn
from .attention import *

class Transformer(nn.Module):

    def __init__(self, 
                 nhead = 8, 
                 dims = [32, 16, 8], 
                 dropout = 0.1):

        super(AttentionFFNBlock, self).__init__()


        self.blocks = [
            AttentionFFNBlock(
                d_model = dim,
                nhead = nhead,
                dim_feedforward = dim * 2,
                dropout = dropout
            ) for dim in dims
        ]
        
        
    def forward(self, x):
        
        
        return x