import torch

class MusicTools:

  """
  Utility class for music theory operations and conversions.
  
  Provides functionality for chord progression analysis, note conversion,
  and music theory operations.

  Attributes:
      device (torch.device): Computation device
      notes (list): List of note names
      chords (dict): Dictionary of chord definitions
      progression_pairs (list): Valid chord progression pairs
      chord_tags (list): List of chord names
      numelem (int): Number of defined chords
      bool_chordnotes (torch.Tensor): Boolean tensor of chord note memberships
      Cp_matrix (torch.Tensor): Chord progression possibility matrix
  """

  def __init__(self, device = None):

    self.device = device if device is None else 'cpu'

    # all music notes
    self.notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # list of the most used chords in tonality Cmajor
    self.chords = {
      "C" : ["C", "E", "G"],
      "C6" : ["C", "E", "G", "A"],
      "D-"  : ["D", "F", "A", "C"],
      "E"   : ["E", "G#", "B", "D"],
      "F-"  : ["F", "Ab", "C", "Eb"],
      "F" : ["F", "A", "C"],
      "Gsus": ["G", "C", "D"],
      "G"   : ["G", "B", "D", "F"],
      "Ab6": ["Ab", "C", "Eb", "F"],
      "A-"  : ["A", "C", "E", "G"],
      "Bb"  : ["Bb", "D", "F","G"],
      "Bdim": ["B", "D", "F", "A"]
    }

    # progression pairs
    self.progression_pairs = [ 
      ["D-", "G"], ["D-", "Bdim"], 
      ["F-", "C"], ["F-", "Ab6"], ["F", "F-"],
      ["F", "G"], ["F", "Bdim"],
      ["Gsus", "G"], ["Gsus", "C"],
      ["G", "C"], ["G", "C6"], ["G", "A-"], ["G", "Ab6"],
      ["Ab6", "Bb"], 
      ["Bb", "C"], ["Bb", "C6"], 
      ["Bdim", "C"], ["Bdim", "C6"],
      ]

    self.chord_tags = list(self.chords.keys())
    self.numelem = len(self.chords)

    self.bool_chordnotes = self.to_bool_chordnotes()
    self.Cp_matrix = self.get_Cp()


  def to_bool_chordnotes(self):
    """
    Convert chord definitions to boolean tensor representation.

    Returns:
        torch.Tensor: Boolean tensor indicating note membership in chords
        every row has True values if and only if the correspondant note name is in the associated chord
    """
    bool_chordnotes = torch.zeros(self.numelem, len(self.notes), dtype = torch.int8, device = self.device)

    for i, chord in enumerate(self.chord_tags):
      values = self.chords[chord]

      for j, value in enumerate(values):

        position = [index for index, note in enumerate(self.notes) if note in self.enharmony(value)][0]
        bool_chordnotes[i, position] = j + 1

    return bool_chordnotes
  
  def to_bool_melody(self, melody):
    """
    Convert melody sequence to boolean tensor representation.

    Args:
        melody (torch.Tensor): Input melody sequence

    Returns:
        torch.Tensor: bool_melody is a boolean tensor of shape (melody_length, number of notes) indicating melody notes
        bool_melody[i, j] = 1 if and only if at time i the note played is j.
    """

    bool_melody = torch.zeros(melody.shape[0], len(self.notes), dtype = torch.int8, device = self.device)
    melody = melody % len(self.notes)

    # every row has all False values but one True, that corresponds to the melody note at the row timestep.
    for i in range(melody.shape[0]): bool_melody[i, melody[i]] = 1

    return bool_melody

  def get_Cp(self):
    """
    Generate chord progression possibility matrix.

    Returns:
        torch.Tensor: Matrix indicating valid chord progressions. Cp[i, j] = 1 iif the pair (chord[i], chord[j]) is a progression pair.
    """

    Cp = torch.zeros(len(self.progression_pairs), len(self.progression_pairs), dtype = torch.int8, device = self.device)
    for pair in self.progression_pairs: Cp[self.chord_tags.index(pair[0]), self.chord_tags.index(pair[1])] = 1
    
    return Cp

  def toNote(self, integers) -> list:
    """
    Convert MIDI note numbers to note names with octaves.

    Args:
        integers (list): List of MIDI note numbers

    Returns:
        list: Note names with octave numbers
    """

    notes = []
    for x in integers: notes.append(f"{self.notes[x % 12]} {x // 12}")

    return notes

  def toChords(self, integers) -> list:
    """
    Convert chord indices to chord names.

    Args:
        integers (list): List of chord indices

    Returns:
        list: Chord names
    """

    chords = []
    for x in integers: chords.append(self.chord_tags[x])

    return chords
  
  def enharmony(self, note):
    """
    Get enharmonic equivalents of a note. (https://en.wikipedia.org/wiki/Enharmonic_equivalence)

    Args:
        note (str): Note name

    Returns:
        list: List of enharmonic equivalents
    """

    if note == "C" or note == "B#": return ["C", "B#"]
    if note == "C#" or note == "Db": return ["C#", "Db"]
    if note == "D": return ["D"]
    if note == "D#" or note == "Eb": return ["D#", "Eb"]
    if note == "E" or note == "Fb": return ["E", "Fb"]
    if note == "F" or note == "E#": return ["F", "E#"]
    if note == "F#" or note == "Gb": return ["F#", "Gb"]
    if note == "G": return ["G"]
    if note == "G#" or note == "Ab": return ["G#", "Ab"]
    if note == "A": return ["A"]
    if note == "A#" or note == "Bb": return ["A#", "Bb"]
    if note == "B" or note == "Cb": return ["B", "Cb"]



if __name__ == '__main__': 

  music_tool = MusicTools()
  melody = torch.tensor([60])

  melody = melody % 12

  c_t = torch.tensor([0, 1, 7, 8])
  c_next = torch.tensor([1, 4, 9, 7])

  print(music_tool.Cp_matrix[c_t, c_next])
  print(music_tool.bool_chordnotes[c_next][:, melody].squeeze())

  print(music_tool.Cp_matrix[c_t, c_next] + music_tool.bool_chordnotes[c_next][:, melody].squeeze())