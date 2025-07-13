import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import NetUtils
from ..utils import create_mlp


class NetAC(NetUtils):

    def __init__(self,
                 
                shared_hidden_dims = [16, 32, 64, 64, 64, 32],
                actor_hidden_dims = [32, 16, 16, 16],
                critic_hidden_dims = [32, 16, 8, 8, 4],

                time_embed = False,
                one_hot_enc = False,
                num_chords : int = 12,
                num_melody : int = 50,
                device : str = 'cpu',
                path = None
            ):
        
        super(NetAC, self).__init__()

        self.num_chords = num_chords
        self.num_melody = num_melody
        self.path = path
        self.type_ = "AC"

        self.t_emb = time_embed
        self.o_hot = one_hot_enc

        input_dim = num_chords + num_melody if self.o_hot else 2
        input_dim += 4 if self.t_emb else 1 

        self.shared = create_mlp(
            input_dim = input_dim,
            hidden_dims = shared_hidden_dims,
        )

        self.actor_head = create_mlp(
            input_dim = shared_hidden_dims[-1],
            hidden_dims = actor_hidden_dims,
            output_dim = num_chords
        )

        self.critic_head = create_mlp(
            input_dim = shared_hidden_dims[-1],
            hidden_dims = critic_hidden_dims,
            output_dim = 1
        )

        self.to(device = device)

        # the initial shared block can't be added here because it can't be used independently. 
        self.blocks = {
            'actor': self.actor_head,
            'critic': self.critic_head
        }

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.shared_params = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        self.actor_params = sum(p.numel() for p in self.actor_head.parameters() if p.requires_grad)
        self.critic_params = sum(p.numel() for p in self.critic_head.parameters() if p.requires_grad)

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.info = f' shared: {self.shared_params} params \n actor head: {self.actor_params} params \n critic head: {self.critic_params} params \n total: {self.total_params}'

    def forward(self, input, tau = 1.0):

        # s_t[:, 0] = c_t, s_t[:, 1] = t, s_t[:, 2] = m_{t + 1}

        chord = input[:, 0]
        melody = input[:, 2]

        time = input[:, 1].unsqueeze(1)

        if self.t_emb: time = self.time_embed(time)

        if self.o_hot: 
            chord = self.one_hot(chord, num_classes = self.num_chords)
            melody = self.one_hot(melody, num_classes = self.num_melody)

        else: chord = chord.unsqueeze(1); melody = melody.unsqueeze(1)

        x = torch.cat([chord, time, melody], dim = 1)
        shared_output = self.shared(x)
        
        logits = self.actor_head(shared_output) / (tau * torch.exp(self.log_temperature))
        actor_output = F.softmax(logits, dim = -1)
        critic_output = self.critic_head(shared_output)

        return {
            "dist": actor_output,
            "values": critic_output[:, 0]
        }