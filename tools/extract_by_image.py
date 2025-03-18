# Extract a spectrogram for every image in the specified folder.
# First run plot_recordings.py to view spectrograms, specifying --seconds 5.
# Delete the spectrogram images you don't want to keep, then run this to import the rest as training data.

import argparse
import inspect
import os
import re
import sys
import time
from pathlib import Path

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import extractor
from core import util

class ExtractByImage(extractor.Extractor):
    def __init__(self, audio_path, images_path, db_name, class_name, class_code):
        super().__init__(audio_path, db_name, class_name, class_code)
        self.images_path = images_path

    # get list of specs from directory of images
    def _process_image_dir(self):
        self.offsets = {}
        for image_path in Path().glob(f"{self.images_path}/*.jpeg"):
            name = Path(image_path).stem
            if '~' in name:
                result = re.split("\\S+~(.+)~.*", name)
                result = re.split("(.+)-(.+)", result[1])
            else:
                result = re.split("(.+)-(.+)", name)
                if len(result) != 4:
                    result = re.split("(\\S+)_(\\S+)", name)

            if len(result) != 4:
                print(f"Error: unknown file name format: {image_path}")
                continue
            else:
                file_name = result[1]
                offset = float(result[2])

            if file_name not in self.offsets:
                self.offsets[file_name] = []

            self.offsets[file_name].append(offset)

    def run(self):
        self._process_image_dir()
        num_inserted = 0
        for recording_path in self.get_recording_paths():
            filename = Path(recording_path).stem
            if filename not in self.offsets:
                continue

            print(f"Processing {recording_path}")
            num_inserted += self.insert_spectrograms(recording_path, self.offsets[filename])

        print(f"Inserted {num_inserted} spectrograms.")

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default=None, help='Class code (required)')
    parser.add_argument('--db', type=str, default='training', help='Database name or full path ending in ".db". Default = "training"')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing recordings (required).')
    parser.add_argument('-i', '--inp', type=str, default=None, help='Directory containing spectrogram images (required).')
    parser.add_argument('--name', type=str, default=None, help='Class name (required)')

    args = parser.parse_args()
    if args.dir is None:
        print("Error: --dir argument is required (directory containing recordings).")
        quit()
    else:
        audio_path = args.dir

    if args.inp is None:
        print("Error: --inp argument is required (directory containing images).")
        quit()
    else:
        image_path = args.inp

    if args.name is None:
        print("Error: --name argument is required (class name).")
        quit()
    else:
        class_name = args.name

    if args.code is None:
        print("Error: --code argument is required (class code).")
        quit()
    else:
        class_code = args.code

    run_start_time = time.time()

    ExtractByImage(audio_path, image_path, args.db, class_name, class_code).run()

    elapsed = time.time() - run_start_time
    print(f'elapsed seconds = {elapsed:.1f}')
