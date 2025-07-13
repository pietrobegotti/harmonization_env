import numpy as np
import torch


class MelodySampler():

    def __init__(self, 
                 device : str,
                 max_length : int = 50,
                 alterations = False):

        self.device = device

        if alterations: self.scale = torch.tensor([0, 2, 4, 5, 7, 8, 9, 10, 11])
        else: self.scale = torch.tensor([0, 2, 4, 5, 7, 9, 11])
        self.notes = torch.tensor(sorted([note + 12 * octave for octave in range(4, 7) for note in self.scale]), 
                                          dtype = torch.int32)
        
        self.probabilities = 1 - torch.arange(0, max_length, dtype = torch.int32) / max_length
        self.probabilities[0] /= 2 # less probability of staying in the same note
        self.probabilities /= torch.sum(self.probabilities)
        
    def get(self,
            start_note : int = 60,
            batch_size : int = 128,
            length : int = 8,
            note_range : int = 5) -> torch.Tensor:
    
        out = torch.zeros((batch_size, length), dtype = torch.float32, device = self.device)
        for i in range(batch_size): out[i, :] = self.get_elem(start_note = start_note, length = length, note_range = note_range)

        return out
    
    def get_new(self,
            batch_size = 512,
            length = 8
        ):

        indices = torch.randint(0, len(self.scale), (batch_size, length))
        return self.scale[indices]

    def get_elem(self,
            start_note : int = 60,
            length : int = 8,
            note_range : int = 5) -> torch.Tensor:


        if start_note % 12 not in self.scale:
            raise ValueError(f"Start note {start_note} is not a valid C Major scale note in the specified octave range.")
        
        probabilities = self.probabilities[0 : note_range] / torch.sum(self.probabilities[0 : note_range])
        movement = 2 * torch.randint(0, 2, size = (length, )) - 1

        samples = torch.multinomial(probabilities, length, replacement = True) * movement

        melody = torch.zeros(length, dtype = torch.int32)
        melody[0] = start_note
        
        note_index = (self.notes == start_note).nonzero(as_tuple = True)[0]

        for i in range(1, length): 

            if note_index + samples[i] < 0 or note_index + samples[i] >= self.notes.shape[0]: note_index -= samples[i]
            else: note_index += samples[i]

            melody[i] = self.notes[note_index]

        return melody
    


if __name__ == '__main__':

    melody_sampler = MelodySampler()
    for _ in range(10000): melody_sampler.get()