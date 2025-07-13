import ffmpeg
import os
from pathlib import Path
from midi2audio import FluidSynth
# from synthviz import create_video
from .create_video import create_video

from ..resources import get_resource_file


class VideoGenerator:

    def __init__(self, 
        filename: str,
        soundfont_path: str,
        wav_file : str = None,
        output_path : str = 'output.mp4'):

        if wav_file is None:
            fs = FluidSynth(soundfont_path)
            fs.midi_to_audio(filename, 'output.wav')
            wav_file = 'output.wav'

        
        abs_path = Path(os.getcwd())
        self.frames_path = str(abs_path / 'video_frames/frame%05d.png')
        self.audio_path = str(abs_path / wav_file)  # Changed from output.wav
        self.output_path = str(abs_path / output_path)
        
        create_video(input_midi = filename)
        # create_video(str(abs_path / filename), video_filename = "output.mp4") 

    def get_video(self, 
                  output_filename : str,
                  audio_filename : str = 'output.wav',
                  input_pattern : str = 'video_frames/frame%05d.png', 
                  framerate : int = 20, 
                  audio_delay : int = 1, 
                  delete_audio : bool = True):
        
        (
            ffmpeg
            .input(input_pattern, framerate = framerate)
            .output('input_video_tmp.mp4', vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run()
        )

        input_video = ffmpeg.input('input_video_tmp.mp4')
        input_audio = ffmpeg.input(audio_filename)
        delayed_audio = input_audio.filter_('adelay', f'{audio_delay*1000}|{audio_delay*1000}')
        
        (
            ffmpeg
            .output(input_video, delayed_audio, output_filename, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run()
        )

        os.remove('input_video_tmp.mp4')
        if delete_audio:
            os.remove(audio_filename)

