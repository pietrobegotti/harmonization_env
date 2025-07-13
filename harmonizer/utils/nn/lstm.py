import torch
import torch.nn as nn
from .create_mlp import create_mlp


class LSTM(nn.Module):

    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 d_model, 
                 enc_hidden_dims, 
                 dec_hidden_dims, 
                 memory_steps_threshold = 4, 
        ):

        super(LSTM, self).__init__()

        self.clock = 0
        self.d_model = d_model
        self.memory_steps = memory_steps_threshold
        self.input_dim = input_dim
        
        self.enc_lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = d_model,
            num_layers = 1,
            batch_first = True
        )

        self.enc_mlp = create_mlp(
            input_dim = d_model,
            hidden_dims = enc_hidden_dims,
            output_dim = latent_dim
        )

        self.dec_mlp = create_mlp(
            input_dim = latent_dim, # + melody_dim,
            hidden_dims = dec_hidden_dims,
            output_dim = input_dim * self.memory_steps
        )

        self.memory = None
        self.h_t = None
        self.c_t = None
        
    def forward(self, z_t):

        if z_t.dim() == 2: z_t = z_t.unsqueeze(1)
        self._memory_update(z_t)

        x_lstm, (h_n, c_n) = self.enc_lstm(z_t, (self.h_t, self.c_t))

        x_lstm = x_lstm[:, -1, :]
        self.h_t = h_n; self.c_t = c_n

        enc_input = h_n.squeeze()
        if enc_input.dim() == 1: enc_input = enc_input.unsqueeze(0)

        latent = self.enc_mlp(enc_input)
        recon = self.dec_mlp(latent)

        if self.clock >= self.memory_steps: delta = recon - torch.flatten(self.memory, 1)   
        else: delta = recon[:, : self.input_dim * self.clock] - torch.flatten(self.memory[:, : self.clock, :], 1)
        
        return latent, delta



    def _memory_update(self, z_t):

        self.memory[:, 1 : , :] = self.memory[:, : -1, :].clone()
        self.memory[:, 0, :] = z_t.squeeze()

        self.clock += 1

    def init_states(self, batch_size, device = None):

        self.h_t = torch.zeros(1, batch_size, self.d_model, device = device)
        self.c_t = torch.zeros(1, batch_size, self.d_model, device = device)
        self.memory = torch.zeros(batch_size, self.memory_steps, self.input_dim)

        self.clock = 0


if __name__ == '__main__':

    batch_size = 32
    input_size = 64
    hidden_size = 128
    
    # Initialize LSTM
    lstm = LSTM(
        input_dim = input_size, 
        d_model = hidden_size, 
        hidden_dims = [128, 64, 12]
    )
    
    # Create random input (batch_size, input_size)
    z_t = torch.randn(batch_size, input_size)
    
    # Initialize hidden state and cell state
    h_t, c_t = lstm.init_states(batch_size)
    
    # Forward pass
    out, h_t_new, c_t_new = lstm(z_t, h_t, c_t)
    
    print(f"Input shape: {z_t.shape}")
    print(f"Hidden state shape: {h_t_new.shape}")
    print(f"Cell state shape: {c_t_new.shape}")
    print(f"out shape: {out.shape}")
