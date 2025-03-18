# Download Google audioset recordings based on a curated list.

import argparse
import inspect
import librosa
import os
from pathlib import Path
import sys
import pandas as pd
import soundfile as sf

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default=None, help='Path to CSV containing curated list.')
parser.add_argument('-o', type=str, default=None, help='Output directory.')
parser.add_argument('-m', type=int, default=1000, help='Only create up to this many 10-second clips (default = 1000).')
parser.add_argument('--sr', type=float, default=cfg.audio.sampling_rate, help=f'Output sampling rate (default = {cfg.audio.sampling_rate}).')

args = parser.parse_args()

if args.i is None or args.o is None:
    print("Error. Both -i and -o arguments are required.")
    quit()

input_path = args.i
output_dir = args.o
max_count = args.m
sample_rate = args.sr

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

curated = pd.read_csv(input_path)
count = 0
for i, row in curated.iterrows():
    youtube_id = row["YTID"]
    start_seconds = row["start_seconds"]

    # download it as wav, which is faster than downloading as mp3;
    # then convert to mp3 when the 10-second clip is extracted
    command = f"yt-dlp -q -o \"{output_dir}/{youtube_id}.%(EXT)s\" -x --audio-format wav https://www.youtube.com/watch?v={youtube_id}"
    print(f"Downloading {youtube_id}")
    os.system(command)

    # extract the 10-second clip and delete the original
    audio_path1 = os.path.join(output_dir, f"{youtube_id}.NA.wav")
    if os.path.exists(audio_path1):
        print(f"Extracting 10-second clip")
        audio_path2 = os.path.join(output_dir, f"{youtube_id}-{int(start_seconds)}.mp3")
        audio, sr = librosa.load(audio_path1, sr=sample_rate)
        start_sample = int(start_seconds * sr)
        end_sample = int((start_seconds + 10) * sr)
        sf.write(audio_path2, audio[start_sample:end_sample], sr, format='mp3')
        os.remove(audio_path1)

        count += 1
        if count >= max_count:
            break

print(f"# downloaded = {count}")
