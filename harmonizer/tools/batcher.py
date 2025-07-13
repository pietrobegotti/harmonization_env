import torch
import torch.nn as nn
from .environment import HarmonizationEnv


class Batcher():

  """
  Handles batch generation and processing for PPO training.
  
  Generates and computes trajectories, advantages estimation, return calculation for the forward pass.

  Attributes:
      device (torch.device): Computation device
      env (HarmonizationEnv): Environment instance
      net (nn.Module): Neural network model
      gamma (float): Discount factor
      lambda_ (float): GAE parameter
      batch_size (int): Size of training batches
      dtype (torch.dtype): Data type for tensors
  """
  
  def __init__(self, device : str, 
               env : HarmonizationEnv, 
               net : nn.Module, 
               gamma : float = 0.99,
               lambda_ : float = 0.99, 
               batch_size : int = 128, 
               dtype = torch.float32):
    """
    Args:
        device (torch.device): Computation device
        env (HarmonizationEnv): Environment instance
        net (nn.Module): Neural network model
        gamma (float, optional): Discount factor. Defaults to 0.99.
        lambda_ (float, optional): GAE parameter. Defaults to 0.99.
        batch_size (int, optional): Batch size. Defaults to 128.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
    """
    
    self.device = device
    self.dtype = dtype
    self.env = env
    self.net = net

    self.batch_size = batch_size
    self.melody_length = None
    self.size = None

    self.gamma = gamma
    self.lambda_ = lambda_

    self.batch_size = batch_size
    self.states, self.actions, self.actions_prob, self.values, self.rewards, self.returns, self.advantages = [None] * 7

    self.setup()

  def setup(self, harmony_init = 0.0):

    """
    Set up tensors for batch processing. Re-initializes them only if needed - i.e. different dimensions - otherwise, 
    when a new batch is created, the same variables are be used as their values are overwritten.

    Args:
        harmony_init (float, optional): Initial harmony value. Defaults to 0.0.
    """
    
    b = self.size == (self.batch_size, self.env.timesteps)
    self.melody_length = self.env.timesteps

    self.size = (self.batch_size, self.melody_length)

    if not b:

      self.size = (self.batch_size, self.melody_length)

      # for every timestep a state is made by (c_t, t, m_{t + 1}) hence the 3
      self.states  = torch.zeros((self.batch_size, self.melody_length, 3), dtype = self.dtype, device = self.device)

      time_indices = torch.arange(self.melody_length)
      time_indices = time_indices.unsqueeze(0).expand(self.size)

      self.states[:, :, 1] = time_indices - 1

    self.states[:, 0, 0] = torch.ones(self.states.shape[0], dtype = torch.int32) * harmony_init
    self.env.clock += 1
    
    # for t in range(1, self.melody_length): self.states[:, t - 1, 2] = self.env.melody[t]
    self.states[:, : self.melody_length - 1, 2] = self.env.melody_mod[:, 1 : self.melody_length]
    
    if not b:

      # self.size = (self.batch_size, self.melody_length)
      self.actions = torch.zeros(self.size, dtype = torch.int32, device = self.device)
      self.values  = torch.zeros(self.size, dtype = self.dtype, device = self.device)
      self.rewards = torch.zeros(self.size, dtype = self.dtype, device = self.device)
      self.actions_prob = torch.zeros(self.size, dtype = self.dtype, device = self.device)
      self.distrib = torch.zeros( (self.batch_size, self.env.timesteps, self.net.num_chords), dtype = self.dtype, device = self.device)
      self.returns = torch.zeros(self.size, dtype = self.dtype, device = self.device)
      self.advantages = torch.zeros(self.size, dtype = self.dtype, device = self.device)

  def get(self, tau = 1.0):
    
    self.setup()
    if self.net.type_ == 'M': self.net.lstm.init_states(batch_size = self.batch_size)

    for k in range(1, self.melody_length):
        
        # for each timestep, generate batchwise the distribution probability over the chords to choose (forward pass) and samples the action.
        # then compute rewards
        
        with torch.no_grad():
          
          output = self.net(self.states[:, k - 1, :], tau = tau)
            
          probs = output["dist"]
          value = output["values"]
          
          self.values[:, k - 1] = value.squeeze()

        next_states = torch.multinomial(probs, 1).squeeze(-1)

        self.states[:, k, 0]   = next_states
        self.actions[:, k - 1] = next_states

        self.actions_prob[:, k - 1] = probs[torch.arange(self.batch_size), next_states]
        self.distrib[:, k - 1, :] = probs
        self.rewards[:, k] = self.env.get_reward(self.states[:, k - 1, :].to(dtype = torch.int32), self.states[:, k, :].to(dtype = torch.int32))

        self.env.clock += 1

    self.returns = self.get_discounted_return()
    self.advantages = self.get_advantages()

    return {
        'returns': self.returns,
        'advantages': self.advantages,
        'states': self.states,
        'actions': self.actions,
        'values': self.values,
        'actions_prob': self.actions_prob,
        'rewards': self.rewards,
        'distrib': self.distrib
    }

  def get_discounted_return(self):

    returns = torch.zeros_like(self.rewards)
    flipped_rewards = torch.flip(self.rewards, dims = [1])

    for i in range(flipped_rewards.shape[1]):
      returns[:, i] = flipped_rewards[:, i - 1] + self.gamma * returns[:, i - 1]

    return torch.flip(returns, dims = [1])
    
  def get_advantages(self):
    
    # Calculate generalized advantage estimates (GAE), as shown in https://arxiv.org/pdf/1707.06347, page 5.
    
    next_values = torch.cat([self.values[:, 1 :], torch.zeros_like(self.values[:, : 1])], dim = 1)
    delta = self.rewards + (self.gamma) * next_values - self.values

    flipped_delta = torch.flip(delta, dims = [1])
    advantages = torch.zeros_like(self.rewards)

    for i in range(1, flipped_delta.shape[1]):
      advantages[:, i] = flipped_delta[:, i] + (self.gamma * self.lambda_) * advantages[:, i - 1]

    return torch.flip(advantages, dims = [1])