import torch.nn as nn

class AttentionFFNBlock(nn.Module):

    def __init__(self, d_model, nhead = 8, dim_feedforward = 2048, dropout = 0.1):

        super(AttentionFFNBlock, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x