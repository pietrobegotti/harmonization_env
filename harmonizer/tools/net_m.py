import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import TransformerAutoencoder, LSTM, create_mlp
from .net_utils import NetUtils

class NetM(NetUtils):

    def __init__(self,
                 
                enc_dim = 32,
                enc_mlp_dims = [32, 32, 32],
                enc_attn_feedforward_dim = 64,
                dec_shared_dims = [32, 64], 
                dec_chord_head_dims = [32, 32, 32],
                dec_melody_head_dims = [32, 32, 32],

                lstm_d_model = 32,
                lstm_enc_hidden_dims = [32, 16],
                lstm_dec_hidden_dims = [16, 32, 64, 32],
                lstm_latent_dim = 8,
                lstm_memory_steps = 4,

                latent_dim = 8,

                actor_hidden_dims : list = [16, 16, 16],
                critic_hidden_dims : list = [16, 8, 4],

                time_embed = False,

                num_chords : int = 12,
                num_melody : int = 50,
                device : str = 'cpu',
                path = None
            ):
        
        super(NetUtils, self).__init__()

        self.num_chords = num_chords
        self.num_melody = num_melody
        self.path = path
        self.latent_dim = latent_dim
        self.t_emb = time_embed

        self.type_ = "M"

        self.ae = TransformerAutoencoder(
            chord_dim = self.num_chords, 
            melody_dim = self.num_melody,
            enc_dim = enc_dim,
            enc_mlp_dims = enc_mlp_dims,
            enc_attn_feedforward_dim = enc_attn_feedforward_dim,
            dec_shared_dims = dec_shared_dims, 
            dec_chord_head_dims = dec_chord_head_dims,
            dec_melody_head_dims = dec_melody_head_dims,
            latent_dim = latent_dim
        )

        self.lstm = LSTM( 
            input_dim = latent_dim, 
            latent_dim = lstm_latent_dim, 
            d_model = lstm_d_model, 
            enc_hidden_dims = lstm_enc_hidden_dims, 
            dec_hidden_dims = lstm_dec_hidden_dims, 
            memory_steps_threshold = lstm_memory_steps, 
        )

        self.actor_head = create_mlp(
            input_dim = latent_dim + lstm_latent_dim,
            hidden_dims = actor_hidden_dims,
            output_dim = num_chords
        )

        self.critic_head = create_mlp(
            input_dim = latent_dim + (4 if self.t_emb else 1),
            hidden_dims = critic_hidden_dims,
            output_dim = 1
        )

        self.to(device = device)

        self.blocks = {
            'ae': self.ae,
            'lstm' : self.lstm,
            'actor': self.actor_head,
            'critic': self.critic_head
        }

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.encoder_params = sum(p.numel() for p in self.ae.encoder_cross.parameters() if p.requires_grad) + sum(p.numel() for p in self.ae.encoder_mlp.parameters() if p.requires_grad)
        self.decoder_params = sum(p.numel() for p in self.ae.decoder.parameters() if p.requires_grad)

        self.lstm_params = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        self.lstm_decoder_params = sum(p.numel() for p in self.lstm.dec_mlp.parameters() if p.requires_grad)
        self.lstm_encoder_params = self.lstm_params - self.lstm_decoder_params

        self.actor_params = sum(p.numel() for p in self.actor_head.parameters() if p.requires_grad)
        self.critic_params = sum(p.numel() for p in self.critic_head.parameters() if p.requires_grad)

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.info = f' encoder: {self.encoder_params} params \n decoder: {self.decoder_params} params \n lstm encoder: {self.lstm_encoder_params} params \n lstm decoder: {self.lstm_decoder_params} params \n actor head: {self.actor_params} params \n critic head: {self.critic_params} params \n total: {self.total_params}'

    def forward(self, input, tau = 1.0):

        # s_t[:, 0] = c_t, s_t[:, 1] = t, s_t[:, 2] = m_{t + 1}

        chord = input[:, 0]
        time = self.time_embed(input[:, 1].unsqueeze(1)) if self.t_emb else input[:, 1].unsqueeze(1)
        melody = input[:, 2]

        # time_embed = self.time_embed(time.unsqueeze(1).clone())

        oh_chord = self.one_hot(chord, num_classes = self.num_chords)
        oh_melody = self.one_hot(melody, num_classes = self.num_melody)


        chord_recon, melody_recon, latent, _ = self.ae(oh_chord, oh_melody)

        lstm_memory, lstm_delta = self.lstm(latent)

        ae_chord_delta = oh_chord - chord_recon
        ae_melody_delta = oh_melody - melody_recon

        actor_input = torch.cat((latent, lstm_memory), dim = 1)
        critic_input = torch.cat((latent, time), dim = 1)

        logits = self.actor_head(actor_input) / (tau * torch.exp(self.log_temperature))
        actor_output = F.softmax(logits, dim = -1)
        critic_output = self.critic_head(critic_input)

        return {
            "dist": actor_output,
            "values": critic_output[:, 0],
            "ae_chord_delta": ae_chord_delta,
            "ae_melody_delta": ae_melody_delta,
            "recon": torch.cat([chord_recon, melody_recon], dim = 1),
            "latent" : latent,
            "lstm_delta" : lstm_delta
        }