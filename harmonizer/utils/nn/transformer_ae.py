import torch
import torch.nn as nn
import torch.nn.functional as F

from .create_mlp import create_mlp

class CrossAttention(nn.Module):

    def __init__(self, chord_dim, melody_dim, hidden_dim):

        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(chord_dim, hidden_dim)
        self.key_proj = nn.Linear(melody_dim, hidden_dim)
        self.value_proj = nn.Linear(melody_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))
        
    def forward(self, chord, melody):
        # chord: (batch_size, n)
        # melody: (batch_size, m)
        
        # Project inputs to query, key, value
        q = self.query_proj(chord)  # (batch_size, hidden_dim)
        k = self.key_proj(melody)   # (batch_size, hidden_dim)
        v = self.value_proj(melody) # (batch_size, hidden_dim)
        
        # Add dimension for attention
        q = q.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        k = k.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        v = v.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, 1, 1)
        
        temperature = torch.exp(self.log_temperature)
        attn_scores = attn_scores / temperature
        
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)  # (batch_size, 1, hidden_dim)
        output = output.squeeze(1)  # (batch_size, hidden_dim)
        
        return output, attn_weights
    
class ResidualMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(ResidualMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):

        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x + residual  
        x = self.norm(x)

        return x
    
class TransformerEncoder(nn.Module):

    def __init__(self, 
                 chord_dim, 
                 melody_dim, 
                 model_dim, 
                 feedforward_dim,
                 kernel_size = 3,
                 padding = 1,
                 stride = 1):

        super(TransformerEncoder, self).__init__()

        self.attention = CrossAttention(chord_dim, melody_dim, model_dim)
        self.residual_mlp = ResidualMLP(model_dim, feedforward_dim)
        
        self.conv1d = nn.Conv1d(
            in_channels = 1, 
            out_channels = 1, 
            kernel_size = kernel_size, 
            padding = padding, 
            stride = stride
        )

    def forward(self, chord, melody):

        attn_output, attn_weights = self.attention(chord, melody)
        mlp_output = self.residual_mlp(attn_output)
        
        # Apply 1D convolution for dimension reduction
        conv_input = mlp_output.unsqueeze(1)  # Add channel dimension (batch_size, 1, hidden_dim)
        conv_output = self.conv1d(conv_input)  # (batch_size, 1, hidden_dim)
        conv_output = conv_output.squeeze(1)   # (batch_size, hidden_dim)
        
        return conv_output, attn_weights

class TransformerDecoder(nn.Module):

    def __init__(self, 
                 latent_dim, 
                 shared_dims, 
                 chord_dims, 
                 melody_dims,
                 out_chord,
                 out_melody):

        super(TransformerDecoder, self).__init__()

        self.shared_mlp = create_mlp(
            input_dim = latent_dim,
            hidden_dims = shared_dims
        ) 

        self.chord_head = create_mlp(
            input_dim = shared_dims[-1],
            hidden_dims = chord_dims,
            output_dim = out_chord 
        )

        self.melody_head = create_mlp(
            input_dim = shared_dims[-1],
            hidden_dims = melody_dims,
            output_dim = out_melody
        )
    
    def forward(self, z):
        
        x = self.shared_mlp(z)
        return self.chord_head(x), self.melody_head(x)

class TransformerAutoencoder(nn.Module):

    def __init__(self, 
                 chord_dim, 
                 melody_dim,
                 enc_dim = 32,
                 enc_mlp_dims = [32, 64, 32, 32],
                 enc_attn_feedforward_dim = 64,
                 dec_shared_dims = [32, 64, 128, 64], 
                 dec_chord_head_dims = [64, 64, 32, 32],
                 dec_melody_head_dims = [64, 64, 32, 32],
                 latent_dim = 16):
        
        super(TransformerAutoencoder, self).__init__()
        
        self.encoder_cross = TransformerEncoder(
            chord_dim = chord_dim, 
            melody_dim = melody_dim, 
            model_dim = enc_dim, 
            feedforward_dim = enc_attn_feedforward_dim
        )

        self.encoder_mlp = create_mlp(
            input_dim = enc_dim,
            hidden_dims = enc_mlp_dims,
            output_dim = latent_dim
        )

        self.decoder = TransformerDecoder(
            latent_dim = latent_dim, 
            shared_dims = dec_shared_dims, 
            chord_dims = dec_chord_head_dims, 
            melody_dims = dec_melody_head_dims,
            out_chord = chord_dim,
            out_melody = melody_dim
        )
        

    def forward(self, chord, melody):

        out, attn_weights = self.encoder_cross(chord, melody)
        latent = self.encoder_mlp(out)
        chord_recon, melody_recon = self.decoder(latent)
        
        return chord_recon, melody_recon, latent, attn_weights
        
# Example usage
if __name__ == "__main__":
    
    
    batch_size = 16
    chord_dim = 24  
    melody_dim = 128

    model = TransformerAutoencoder(
        chord_dim=chord_dim,
        melody_dim=melody_dim,
        enc_dim=32,
        feedforward_dim=64,
        dec_shared_dims=[32, 64, 128, 64],
        dec_chord_head_dims=[64, 64, 32, 32],
        dec_melody_head_dims=[64, 64, 32, 32],
        latent_dim = 16
    )

    melody = torch.randn(batch_size, melody_dim)
    chord = torch.randn(batch_size, chord_dim)

    model(chord, melody)
    
