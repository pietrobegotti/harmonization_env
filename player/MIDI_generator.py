import torch
import pygame
from midiutil import MIDIFile

class MIDIGenerator:

    def __init__(self, 
                 tempo: int = 60, 
                 volume: int | list[int] = 100, 
                 channel: int = 0):
        
        self.tempo = tempo
        self.volume = volume if isinstance(volume, list) else [volume]
        self.channel = channel
        
        self.voices = {}
        pygame.mixer.init()

    def generate(self, 
                voices: torch.Tensor, 
                filenames: str | list[str] = "output.mid") -> str:
        
        for i in range(voices.shape[0]):

            notes = [voices[i, 0]]; len = [4.0]

            for j in range(1, voices.shape[1]):

                x = voices[i, j]
                if x == notes[-1]: len[-1] += 4.0
                else:
                    notes.append(x)
                    len.append(4.0)
            
            len[-1] += 16.0

            self.voices[i] = {
                'notes' :  notes,
                'len' : len
            }

        # self._create_combined_midi_file(filenames)
        self._alternative_generate(filenames)
     
    def _create_combined_midi_file(self, filename):
        """Create a combined MIDI file with all voices playing simultaneously"""
        # Create a MIDI file with as many tracks as voices
        num_tracks = len(self.voices)
        midi = MIDIFile(num_tracks)
        
        # Process each voice
        for voice_idx in range(num_tracks):

            track = voice_idx
            time = 0
            volume_idx = min(voice_idx, len(self.volume) - 1)
            
            # Set tempo for this track
            midi.addTempo(track, time, self.tempo)
            
            # Add notes for this voice
            voice_data = self.voices[voice_idx]
            notes = voice_data['notes']
            lens = voice_data['len']
            
            for note, duration in zip(notes, lens):
                if note <= 0:
                    time += duration / 4
                    continue
                    
                # Add note to this track
                midi.addNote(track, voice_idx, int(note), time, duration / 4, 
                             self.volume[volume_idx])
                time += duration / 4
        
        # Write the combined file
        with open(filename, "wb") as f: midi.writeFile(f)

    def _alternative_generate(self, filename):
        
        """Create a MIDI file with all voices compressed into a single track"""
        # Create a MIDI file with a single track
        midi = MIDIFile(1)
        track = 0
        
        # Set tempo for the track
        midi.addTempo(track, 0, self.tempo)
        
        # Process each voice
        for voice_idx in range(len(self.voices)):
            voice_data = self.voices[voice_idx]
            notes = voice_data['notes']
            lens = voice_data['len']
            
            # Current time position for this voice
            time = 0
            volume_idx = min(voice_idx, len(self.volume) - 1)
            
            for note, duration in zip(notes, lens):
                if note <= 0:
                    time += duration / 4
                    continue
                    
                # Add note to the single track but use different channels for different voices
                midi.addNote(track, voice_idx, int(note), time, duration / 4, 
                             self.volume[volume_idx])
                time += duration / 4
        
        # Write the file
        with open(filename, "wb") as f: midi.writeFile(f)

    def play(self, filename: str):

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy(): pygame.time.wait(100)

        except Exception as e: print(f"Error playing MIDI file: {e}")