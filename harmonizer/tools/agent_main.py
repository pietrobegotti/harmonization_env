import random
import torch
import torch.nn as nn
import time
import copy

from tqdm import tqdm

from .agent_utils import AgentUtils
from .environment import HarmonizationEnv
from ..utils import MusicTools

class Agent(AgentUtils):

  def __init__(self, 
               env : HarmonizationEnv, 
               net : nn.Module, 
               **kwargs):

    super().__init__(env, net, **kwargs)

  def train(
      self, 
      num_iterations: int = 1000, 
      iter_per_batch: int = 5, 
      check_step: int = 10, 
      fine_tune: bool = False,
      save: bool = True,
      print_out: bool = True,
      track_progress = False,
      blocks = None
    ):
    
    if blocks is None: blocks = ['full']

    if track_progress: 
      progress = { 
        "block" : [], 
        "iter" : [], 
        "ratio" : [], 
        "avg_reward" : [], 
        "time" : [],
        "avg_grad_norm" : [],
        "loss" : {'actor' : [], 'critic' : [], 'entropy' : [], 'ae' : [], 'lstm' : [], 'full' : []}
      }

    if num_iterations < check_step: check_step = 1

    critic_loss_fn = nn.HuberLoss()
    iterator = range(num_iterations) if print_out else tqdm(range(num_iterations))

    for num_iter in iterator:

      timesteps = random.randint(8, 32)

      if fine_tune: self.env.reset()
      else: self.env.reset(melody = self.sampler.get_new(batch_size = self.batcher.batch_size, length = timesteps), timesteps = timesteps)

      data = self.batcher.get()

      for key, value in data.items(): setattr(self, key, value)
      self.actions_prob = self.actions_prob.detach().clone()
      iter_grad_norm  = 0
      
      avg_iter_time = 0

      for _ in range(iter_per_batch):

        init_time = time.time()

        new_probs, critic_values, avg_entropies, avg_ae_error, avg_out, lstm_err = self._initialize_tensors(timesteps)
        
        block = random.choice(blocks)
        self.optimizer = self.train_specific_block(blocks_to_train = block)

        self.optimizer.zero_grad()
        if self.net.type_ == 'M': self.net.lstm.init_states(batch_size = self.batch_size)

        for k in range(1, timesteps):
  
            (
              critic_values[:, k],
              new_probs[:, k - 1],
              avg_entropies[k - 1],
              avg_ae_error[k - 1],
              avg_out[k - 1],
              lstm_err[k - 1]
            ) = self.forward(k)

        ratio = 1 + torch.log(new_probs + 1e-10) - torch.log(self.actions_prob + 1e-10)  # new_probs / (self.actions_prob + 1e-3)
        ratio[:, -1] = 1
        
        weighted_probs = self.advantages * ratio
        weighted_clipped_probs = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantages

        losses = {
          'actor' : - torch.min(weighted_probs, weighted_clipped_probs).mean(),
          'critic' : critic_loss_fn(critic_values, self.returns),  
          'entropy' : - avg_entropies.mean(),
          'ae' : avg_ae_error.mean() + 0.5 * avg_out.mean(),
          'lstm' : lstm_err.mean()
        }

        loss_full = 0
        for key in losses: loss_full += self.net_weights[key] * losses[key]
        losses['full'] = loss_full

        if block == 'actor': loss = self.net_weights['actor'] * losses['actor'] + self.net_weights['entropy'] * losses['entropy']
        else: loss = self.net_weights[block] * losses[block]

        if iter_per_batch > 1: loss.backward(retain_graph = True)
        else: loss.backward()

        grad_norm = 0.0
        count = 0

        for param in self.net.parameters():
            if param.grad is not None: grad_norm += param.grad.data.norm(1).item(); count += 1

        avg_grad_norm = grad_norm / sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.scale_gradients()
        iter_grad_norm += avg_grad_norm

        self.optimizer.step()
        avg_iter_time += time.time() - init_time
      
      avg_iter_time /= iter_per_batch
      iter_grad_norm /= iter_per_batch

      if num_iter % check_step == 0:

        self.learning_rates[block] *= self.lr_decay
       
        if print_out: 
          print(f"n_iter: {num_iter}/{num_iterations}, grad norm: {iter_grad_norm:.3f}. Training block: {block}") # loss: {losses['total']:.3f}
          print(f"actor: {losses['actor']:.3f}, critic: {losses['critic']:.3f}, entropy: {losses['entropy']:.3f}, ae: {losses['ae']:.3f}, lstm: {losses['lstm']:.3f}")
          print(f"Average reward: {self.rewards.mean():.3f}")
          print()

        if track_progress:
          progress['block'].append(block)
          progress['iter'].append(num_iter)
          progress['ratio'].append(ratio.mean().item())
          progress['avg_reward'].append(self.rewards.mean().item())
          progress['time'].append(avg_iter_time)
          progress['avg_grad_norm'].append(avg_grad_norm)
          for key in progress['loss']: progress['loss'][key].append(losses[key].item())

        if save and self.net.path is not None: self.net.save()

    if track_progress: return progress

  def get(self, 
          melody : torch.Tensor,
          fine_tune : bool = False,
          num_iterations : int = 1000,
          num_samples : int = 20000,
          tau : int = 0.5,
          top_k : int = 1):
    
    self.net.eval()

    if fine_tune: 
        
        self.train(
            num_iterations = num_iterations, 
            fine_tune = True, 
            save = False,
            print_out = False,
            blocks = ['actor', 'critic']
        )

    samples = self.sampler.get(batch_size = self.batcher.batch_size, length = melody.shape[-1])
    samples[ : num_samples, :] = melody

    self.env.reset(melody = samples, timesteps = melody.shape[-1])
    with torch.no_grad():

      data = self.batcher.get(tau = tau)
      for key, value in data.items(): setattr(self, key, value)
        
    total_rewards = torch.sum(self.rewards, dim = 1)
    top_indices = torch.topk(total_rewards, min(top_k, len(total_rewards))).indices

    top_states = self.states[top_indices, :, 0].to(dtype = torch.int32)
    top_rewards = self.rewards[top_indices, 1 :]

    avg_reward = self.rewards[:, 1 : ].mean()
    var_reward = self.rewards[:, 1 : ].var()

    # print(f'melody avg reward: {self.rewards[0, 1 : ].mean():.3f}')
    print(f'rewards. mean : {avg_reward:.3f}, var : {var_reward:.3f}')

    self.net.train()
    
    return top_states, top_rewards
  
  def test(
      self, 
      num_samples: int = 1000, 
    ):
    
    with torch.no_grad():

      original_batch_size = self.batcher.batch_size
      self.batcher.batch_size = num_samples

      timesteps = random.randint(8, 32)

      melodies = self.sampler.get(batch_size = self.batcher.batch_size, length = timesteps)
      self.env.reset(melody = melodies, timesteps = timesteps)

      data = self.batcher.get()
      for key, value in data.items(): setattr(self, key, value)
       
    self.batcher.batch_size = original_batch_size

    avg_reward = self.rewards[:, 1 : ].mean()
    var_reward = self.rewards[:, 1 : ].var()

    print(f'rewards. mean : {avg_reward:.3f}, var : {var_reward:.3f}')

  def onestep_forward(self, next_melody, chord, time = 0, tau = 1.0):

    self.net.eval()
    if self.net.type_ == 'M': self.net.lstm.init_states(batch_size = 1) # self.batch_size)

    mt = MusicTools()

    if isinstance(next_melody, str): next_melody = mt.notes.index(next_melody)
    if isinstance(chord, str): chord = mt.chord_tags.index(chord)

    state = torch.tensor((chord, time, next_melody), dtype = torch.float32, device = self.device)
    if state.ndim == 1: state = state.unsqueeze(0)

    output = self.net(state, tau = tau)
    rewards = []

    for i in range(self.net.num_chords):
      s_next = torch.tensor((i, time + 1, 0), dtype = torch.int32, device = self.device).unsqueeze(0)
      reward = self.env.get_reward(s_t = state.to(dtype = torch.int32), s_next = s_next)

      rewards.append(reward.item())

    return output, rewards
