import torch
import torch.nn as nn
import torch.nn.functional as F

from .environment import HarmonizationEnv
from .batcher import Batcher
from ..utils import MelodySampler


class AgentUtils:

  def __init__(self, 
               env : HarmonizationEnv, 
               net : nn.Module, 
               **kwargs):

    self.env = env
    self.device = self.env.device

    self.net = net
    self.net = net.to(self.device)
  
    self.batch_size = kwargs.get('batch_size', 128)
    self.grad_scale_threshold = kwargs.get('grad_scale', 1.0)
  
    self.gamma = kwargs.get('gamma', 0.5)
    self.lambda_ = kwargs.get('lambda_', 0.95)
    self.lr = kwargs.get('lr', 1e-4)
    self.lr_decay = kwargs.get('lr_decay', 1 - 1e-5)
    self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.net.parameters(), lr = self.lr))

    self.epsilon = kwargs.get('epsilon', 0.2)
    self.batcher = kwargs.get('batcher', None)

    self.load_ = kwargs.get('load', False)
    self.alterations = kwargs.get('alterations', False)

    self.path = self.net.path # self.path = kwargs.get('path', 'net_mlp.pth')

    self.sampler = MelodySampler(device = self.device, alterations = self.alterations)

    if self.batcher is not None: self.batcher = self.batcher 

    else: self.batcher = Batcher(
        device = self.device, 
        env = self.env, 
        net = self.net, 
        gamma = self.gamma, 
        lambda_ = self.lambda_, 
        batch_size = self.batch_size
    )

    if self.load_: self.load()

    self.net_weights = {
      'actor' : kwargs.get('actor_weight', 1.0),
      'critic' : kwargs.get('critic_weight', 0.5),
      'ae' : kwargs.get('ae_weight', 1.0),
      'entropy' : kwargs.get('entropy_weight', 0.1),
      'lstm' : kwargs.get('lstm_weight', 1.0),
      'full' : 1.0
    }

    self.learning_rates = kwargs.get('learning_rates', {
      'critic': 1e-4,
      'actor': 1e-4,
      'full' : 1e-4,
    })

    if self.net.type_ == 'AE':
      self.learning_rates['ae'] = 1e-4
    if self.net.type_ == 'M':
      self.learning_rates['ae'] = 1e-4; self.learning_rates['lstm'] = 1e-4

  def train_specific_block(self, blocks_to_train):

    if blocks_to_train == 'full': 
      for param in self.net.parameters(): param.requires_grad = True
      return torch.optim.Adam(self.net.parameters(), lr = self.learning_rates[blocks_to_train], weight_decay = 1e-5)

    for param in self.net.parameters(): param.requires_grad = False
    param_groups = []

    net_block = self.net.blocks[blocks_to_train]
    lr_block = self.learning_rates[blocks_to_train]

    if not isinstance(net_block, list): net_block = [net_block]
    
    for block in net_block:
      for param in block.parameters(): param.requires_grad = True
      param_groups.append({'params': block.parameters(), 'lr': lr_block})

    optimizer = torch.optim.Adam(param_groups)
    return optimizer

  def scale_gradients(self):

    parameters = [p for p in self.net.parameters() if p.grad is not None and p.requires_grad]
    if not parameters: return 0.0
    
    grad_norms = [torch.norm(p.grad.detach()).item() for p in parameters]

    if len(grad_norms) > 100:  
        grad_norms.sort()
        max_norm = grad_norms[int(len(grad_norms) * 0.999)]

    else: max_norm = max(grad_norms)
    
    scale_factor = min(1.0, self.grad_scale_threshold / (max_norm + 1e-6))
    
    for param in parameters: param.grad.mul_(scale_factor)
    
  def _initialize_tensors(self, timesteps):

    new_probs = torch.zeros((self.batch_size, timesteps), dtype = torch.float32, device = self.device)
    critic_values = torch.zeros((self.batch_size, timesteps), dtype = torch.float32, device = self.device)
    avg_entropies = torch.zeros(timesteps - 1, dtype = torch.float32, device = self.device)

    avg_ae_error = torch.zeros(timesteps - 1, dtype = torch.float32, device = self.device)
    avg_out = torch.zeros(timesteps - 1, dtype = torch.float32, device = self.device)

    lstm_error = torch.zeros(timesteps - 1, dtype = torch.float32, device = self.device)

    return new_probs, critic_values, avg_entropies, avg_ae_error, avg_out, lstm_error

  def forward(self, k):

    critic_value, new_prob, avg_entropy, avg_ae_error, avg_out, lstm_err = 0, 0, 0, 0, 0, 0

    output = self.net(self.states[:, k - 1, :])
    
    new_dist = output["dist"]
    new_values = output["values"]

    critic_value = new_values.squeeze()
    new_prob = new_dist[torch.arange(self.batch_size), self.actions[:, k - 1]]
    avg_entropy = - torch.sum(new_dist * torch.log(new_dist + 1e-10), dim = 1).mean()

    if self.net.type_ == 'AE' or self.net.type_ == 'M':
    
      ae_chord_delta = output["ae_chord_delta"]
      ae_melody_delta = output["ae_melody_delta"]
      out = output["recon"]

      avg_ae_error = 0.5 * torch.mean(ae_chord_delta ** 2) + 0.5 * torch.mean(ae_melody_delta ** 2)
      avg_out = (0.5 - 0.5 * torch.cos(2 * torch.pi * out)).mean()
    
    if self.net.type_ == 'M':
      
      lstm_delta = output["lstm_delta"] 
      lstm_err = F.mse_loss(lstm_delta, target = torch.zeros_like(lstm_delta))

    return critic_value, new_prob, avg_entropy, avg_ae_error, avg_out, lstm_err

  def save(self, path : str = None):
    
    if path == None: path = self.path 
    torch.save(self.net.state_dict(), path)

  def load(self):

      net_dict = torch.load(self.path, weights_only = False, map_location = torch.device('cpu'))
      self.net.load_state_dict(net_dict)