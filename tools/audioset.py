# Given a Google Audioset class name (e.g. Wind) and output directory,
# download the corresponding recordings from Youtube, extract the
# relevant 10-second segments and create audio files in the output
# directory.

# The folder data/audioset folder contains the relevant metadata.
# See class_label_indices.csv for the list of classes, such as "Wind"
# and "Sheep". However, most clips have multiple labels. If you want
# recordings of sheep, secondary labels such as "Bleat" and "Animal" are
# acceptable, but other such as "Music" may not be. The file class_inclusion.csv
# allows you to specify up to 10 allowable secondary labels for a class.
# If you request a class that is not defined in class_inclusion.csv,
# only clips with no secondary labels will be used, which will likely omit a lot
# of useful clips.

# To get a report on what secondary labels are associated with a specified class,
# specify the -r flag, which will report on labels and not download any recordings.

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
parser.add_argument('-c', type=str, default=None, help='Google Audioset class name (e.g. "Wind").')
parser.add_argument('-o', type=str, default=None, help='Output directory.')
parser.add_argument('-m', type=int, default=1000, help='Only create up to this many 10-second clips (default = 1000).')
parser.add_argument('-r', action="store_true", help='If specified, just report on secondary classes for the specified class.')
parser.add_argument('--skip', type=int, default=0, help='Skip this many initial recordings.')
parser.add_argument('--sr', type=float, default=cfg.audio.sampling_rate, help=f'Output sampling rate (default = {cfg.audio.sampling_rate}).')

args = parser.parse_args()

do_report = args.r
if args.c is None:
    print("Missing required -c argument.")
    quit()

if not do_report and args.o is None:
    print("Missing required -o argument.")
    quit()

class_name = args.c
output_dir = args.o
max_count = args.m
num_to_skip = args.skip
sample_rate = args.sr

if not do_report and not os.path.exists(output_dir):
    os.mkdir(output_dir)

# read class info
class_label_path = Path("..") / Path("data") / Path("audioset") / Path("class_list.csv")
df = pd.read_csv(str(class_label_path))
name_to_index = {}
index_to_label = {}
label_to_name = {}
for i, row in df.iterrows():
    name_to_index[row["display_name"]] = row["index"]
    index_to_label[row["index"]] = row["mid"]
    label_to_name[row["mid"]] = row["display_name"]

if class_name not in name_to_index:
    print(f"Class \"{class_name}\" not found. See names in \"{class_label_path}\".")
    quit()

class_index = name_to_index[class_name]
class_label = index_to_label[class_index]

# read info for all clips that match the specified class
details_path = Path("..") / Path("data") / Path("audioset") / Path("unbalanced_train_segments.csv")
df = pd.read_csv(str(details_path), quotechar='"', skipinitialspace=True, low_memory=False)
label_counts = {}
num_unique = 0 # number with no other labels
class_rows = []
for i, row in df.iterrows():
    labels = row["positive_labels"].split(',')
    if class_label in labels:
        class_rows.append((row["YTID"], row["start_seconds"], labels))
        if len(labels) == 1:
            num_unique += 1

        for label in labels:
            if label == class_label:
                continue

            if label not in label_counts:
                label_counts[label] = 0

            label_counts[label] += 1

if do_report:
    print(f"# segments with no secondary labels = {num_unique}")
    print()
    for label in label_counts:
        print(f"# segments also labelled {label_to_name[label]} = {label_counts[label]}")

    quit()

# get any allowable secondary labels
class_inclusion_path = Path("..") / Path("data") / Path("audioset") / Path("class_inclusion.csv")
df = pd.read_csv(str(class_inclusion_path))
allowed_labels = set([class_label])
for i, row in df.iterrows():
    if row["Name"] == class_name:
        for i in range(1, 11):
            label = row[f"Include{i}"]
            if not pd.isna(label):
                if not label in name_to_index:
                    print(f"Error: value \"{label}\" in class_inclusion.csv is not a known class name.")
                    quit()

                allowed_labels.add(index_to_label[name_to_index[label]])

# download recordings and save only the relevant 10-second segment of each
count = 0
for youtube_id, start_seconds, labels in class_rows:
    if all(label in allowed_labels for label in labels):
        if count < num_to_skip:
            count += 1
            continue

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
            if count >= max_count + num_to_skip:
                break

print(f"# downloaded = {count - num_to_skip}")
