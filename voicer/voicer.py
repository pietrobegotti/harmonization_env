import torch
from typing import Tuple, List
from ..harmonizer import MusicTools
from ..player import MIDIGenerator

class Voicer:
    def __init__(self,
                 melody: torch.Tensor,
                 chords: torch.Tensor,
                 n_voices: int = 4,
                 range_: Tuple[int, int] = (36, 84)) -> None:
        
        self.melody = melody.to(dtype = torch.int32)
        self.chords = chords
        self.range = range_
        self.n_voices = n_voices
        self.voices = torch.zeros(self.n_voices, melody.shape[0], dtype=torch.int32)
        self.voices[-1, :] = self.melody
        self.tools = MusicTools()
        self.chords_notes = self.tools.bool_chordnotes[self.chords]

    # for the moment the bass is just the fundamental of the chord
    def get_bass(self): return torch.nonzero(self.chords_notes == 1, as_tuple = True)


    def get_inner(self, num_iter: int = 200) -> torch.Tensor:

        voices = self.voices.clone()
        available = torch.nonzero(self.chords_notes != 0)
        max_index = available[:, 0].max().item()

        tensors = [available[available[:, 0] == i, 1] for i in range(max_index + 1)]

        for t, tensor in enumerate(tensors):   
            max_reward = float('-inf')

            # to generate inner voices there's some random sampling over all the possible combinations, and computing the reward 
            # for every sample.

            for _ in range(num_iter):

                indices = torch.randperm(tensor.shape[0])[:2]
                octaves = torch.randint(1, 3, (self.n_voices - 2,), dtype=torch.int32)
                
                inner_voices = torch.sort(tensor[indices] + self.range[0] + 12 * octaves)[0]

                if inner_voices[0] < voices[0, t] or inner_voices[-1] > voices[-1, t]: continue

                voices[1 : self.n_voices - 1, t] = inner_voices
                reward = self.get_reward(voices, t)

                if reward > max_reward:
                    max_reward = reward
                    self.voices[1:self.n_voices - 1, t] = inner_voices
        
        return self.voices


    def get(self) -> torch.Tensor:

        rows, bass = self.get_bass()

        self.chords_notes[rows, bass] = 0
        self.voices[0, :] = bass + self.range[0]

        x = self.voices[0, 0]

        for i in range(1, self.voices.shape[1]):
            x_next = self.voices[0, i]

            ref = torch.abs(x - x_next)

            if torch.abs(x - (x_next + 12)) < ref: x_next += 12
            elif torch.abs(x - (x_next - 12)) < ref: x_next -= 12
            
            if x_next < self.range[0]: x_next += 12
            
            self.voices[0, i] = x_next
            x = x_next
        
        melody_mask = self.tools.to_bool_melody(self.melody).to(dtype = torch.bool)
        self.chords_notes[melody_mask] = 0
        
        self.voices = self.get_inner()
        return self.voices

    def get_reward(self, voices: torch.Tensor, t: int) -> float:

        reward = 0.0
        current_voices = voices[:, t]
        intervals = current_voices[1:] - current_voices[:-1] 

        if t > 0:

            past_voices = voices[:, t - 1]
            past_intervals = past_voices[1:] - past_voices[:-1]
            delta = intervals - past_intervals

            if torch.any((intervals == 6) | (intervals == 11)): reward -= 1.0
            if torch.any((intervals == 7) & (delta == 0)): reward -= 1.0

            reward -= (torch.abs(current_voices[1] - past_voices[1]) +
                      torch.abs(current_voices[2] - past_voices[2])) / 5 # voices moving the least between past and current note will be rewarded more
            

        # to provide some space between voices, especially wrt the upper and lower most.
        reward += intervals[0].item() / 4 + intervals[-1].item() / 8
        for i in range(1, intervals.shape[0] - 1): reward += intervals[i].item() / 10

        return reward

if __name__ == '__main__':

    melody = torch.tensor([60, 62, 64, 65, 67, 69, 71, 72], dtype = torch.int32)
    melody += 12
    chord_tags = ['C6', 'G', 'A-', 'D-', 'Gsus', 'A-', 'G', 'C']
    tools = MusicTools()
    
    # Create chords tensor
    positions = torch.tensor([tools.chord_tags.index(elem) for elem in chord_tags], dtype=torch.int32)
    
    voicer = Voicer(melody, positions)
    voices = voicer.get()

    player = MIDIGenerator()
    player.generate(voices, filenames = 'test0.mid')
    player.play(filename = 'test0.mid')