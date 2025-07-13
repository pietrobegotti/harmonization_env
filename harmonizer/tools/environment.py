import numpy as np
import torch

from ..utils import *

class HarmonizationEnv:

    def __init__(self, melody = None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 alpha = 0.6, harmony_init = 0.0):

        super(HarmonizationEnv, self).__init__()

        self.device = device
        self.H = MusicTools()

        if melody is None: melody = [0] * 8

        if isinstance(melody, list): self.melody = torch.tensor(melody, dtype = torch.float32, device = self.device)
        elif isinstance(melody, np.ndarray): self.melody = torch.from_numpy(melody, dtype = torch.float32, device = self.device)
        elif isinstance(melody, torch.Tensor): self.melody = melody.to(dtype = torch.float32, device = self.device)

        if self.melody.dim() == 1: self.melody = torch.unsqueeze(self.melody, dim = 0)

        new_col = torch.full((self.melody.shape[0], 1), -1, device = self.device)
        self.melody = torch.cat([new_col, self.melody], dim = 1) 
        # self.melody = torch.cat([torch.tensor([-1], dtype = torch.float32, device = self.device), self.melody])

        self.melody_mod = self.melody % 12
        self.melody_mod[:, 0] = -1

        self.action_space_dim = self.H.numelem

        self.harmony_init = harmony_init
        self.clock = 0
        self.alpha = alpha
        self.timesteps = self.melody.shape[-1]


    def reset(self, melody = None, timesteps = None, init_state = None, alpha = None):

        self.clock = 0

        if melody is not None:

            if isinstance(melody, list): self.melody = torch.tensor(melody, dtype = torch.float32, device = self.device)
            elif isinstance(melody, np.ndarray): self.melody = torch.from_numpy(melody, dtype = torch.float32, device = self.device)
            elif isinstance(melody, torch.Tensor): self.melody = melody.to(dtype = torch.float32, device = self.device)
            
            if self.melody.dim() == 1: self.melody = torch.unsqueeze(self.melody, dim = 0)
            
            new_col = torch.full((self.melody.shape[0], 1), -1, device = self.device)
            self.melody = torch.cat([self.melody, new_col], dim = 1) 

            # self.melody = torch.cat([torch.tensor([-1], dtype = torch.float32, device = self.device), self.melody])

            self.melody_mod = self.melody % 12
            self.melody_mod[:, 0] = -1
            
        if init_state is not None:  self.harmony_init = init_state
        if alpha is not None: self.alpha = alpha
        if timesteps is not None: self.timesteps = timesteps
        
        return self.harmony_init


    def get_reward(self, s_t, s_next):

        # s_t is 2D - (batch_size x 3)
        # s_t[:, 0] = c_t, s_t[:, 1] = t, s_t[:, 2] = m_{t + 1}

        melody_reward = torch.diagonal(
            self.H.bool_chordnotes[s_next[:, 0]][:, s_t[:, 2]]
        )
        
        if s_t[0, 0] == -1: chord_progression_reward = 0
        else: chord_progression_reward = self.H.Cp_matrix[s_t[:, 0], s_next[:, 0]]

        return self.alpha * melody_reward.ne(0) + (1 - self.alpha) * chord_progression_reward

    
if __name__ == '__main__': 

    h = HarmonizationEnv(melody = [60, 62, 64])

    s_t = torch.tensor([[3, 5, 8],
        [0, 7, 4]])
    s_next = torch.tensor([[3, 5, 8],
        [0, 7, 4]])

    h.get_reward(s_t, s_next)
