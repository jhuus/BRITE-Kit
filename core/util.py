# Utility functions

import glob
import os
import re
from pathlib import Path
from posixpath import splitext
from types import SimpleNamespace
import zlib

import numpy as np

from core import cfg
from core import database

AUDIO_EXTS = [
  '.3gp', '.3gpp', '.8svx', '.aa', '.aac', '.aax', '.act', '.aif', '.aiff', '.alac', '.amr', '.ape', '.au',
  '.awb', '.cda', '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.m4p', '.mmf',
  '.mp3', '.mpc', '.mpga', '.msv', '.nmf', '.octet-stream', '.ogg', '.oga', '.mogg', '.opus', '.org',
  '.ra', '.rm', '.raw', '.rf64', '.sln', '.tta', '.voc', '.vox', '.wav', '.wma', '.wv', '.webm', '.x-m4a',
]

# compress a spectrogram in preparation for inserting into database
def compress_spectrogram(data):
    data = data * 255
    np_bytes = data.astype(np.uint8)
    bytes = np_bytes.tobytes()
    compressed = zlib.compress(bytes)
    return compressed

# decompress a spectrogram, then convert from bytes to floats and reshape it
def expand_spectrogram(spec):
    bytes = zlib.decompress(spec)
    spec = np.frombuffer(bytes, dtype=np.uint8) / 255
    spec = spec.astype(np.float32)

    spec = spec.reshape(cfg.audio.spec_height, cfg.audio.spec_width, 1)
    return spec

def format_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"{hours:02}H:{minutes:02}M:{seconds:02}S"

# return list of audio files in the given directory;
# returned file names are fully qualified paths, unless short_names=True
def get_audio_files(path, short_names=False):
    files = []
    if os.path.isdir(path):
        for file_name in sorted(os.listdir(path)):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                base, ext = os.path.splitext(file_path)
                if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
                    if short_names:
                        files.append(file_name)
                    else:
                        files.append(file_path)

    return sorted(files)

# If db_name ends in ".db", treat it as a full path.
# Otherwise use the path "../data/{db_name}.db"
def get_database(db_name):
    if db_name.endswith('.db'):
        db = database.Database(db_name)
    else:
        db = database.Database(f"../data/{db_name}.db")

    return db

# return list of strings representing the lines in a text file,
# removing leading and trailing whitespace and ignoring blank lines
# and lines that start with #
def get_file_lines(path):
    try:
        with open(path, 'r') as file:
            lines = []
            for line in file.readlines():
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    lines.append(line)

            return lines
    except IOError:
        print(f'Unable to open input file {path}')
        return []

# return the version number, as defined in version.txt, if it exists
def get_version(file_path="version.txt"):
    if os.path.exists(file_path):
        lines = get_file_lines(file_path)
        if len(lines) > 0:
            return lines[0]

    return ""

# return True iff given path is an audio file
def is_audio_file(file_path):
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_path)
        if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
            return True

    return False
