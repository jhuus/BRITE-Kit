# Extract a spectrogram at every n (default 5) seconds of every recording in the specified folder.
# This populates a database that can be searched for suitable training data.

import argparse
import inspect
import os
import sys
import time
from pathlib import Path
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import extractor

class ExtractAll(extractor.Extractor):
    def __init__(self, audio_path, db_name, class_name, class_code, increment, max_num, max_per):
        super().__init__(audio_path, db_name, class_name, class_code)
        self.increment = increment
        self.max_num = max_num
        self.max_per = max_per

    def run(self):
        num_inserted = 0
        for recording_path in self.get_recording_paths():
            filename = Path(recording_path).stem
            if filename in self.filenames:
                continue # don't process ones that exist in database already

            print(f"Processing {recording_path}")
            try:
                seconds = self.load_audio(recording_path)
            except Exception as e:
                print(f"Caught exception: {e}")
                continue

            if seconds < self.increment:
                continue # recording is too short

            offsets = np.arange(0, seconds - self.increment, self.increment)
            if self.max_per is not None:
                offsets = offsets[:self.max_per]

            if self.max_num is not None and num_inserted + len(offsets) > self.max_num:
                num_needed = self.max_num - (num_inserted + len(offsets))
                offsets = offsets[:num_needed]

            num_inserted += self.insert_spectrograms(recording_path, offsets)
            if self.max_num is not None and num_inserted >= self.max_num:
                break

        print(f"Inserted {num_inserted} spectrograms.")

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default=None, help='Class code (required).')
    parser.add_argument('--db', type=str, default='training', help='Database name. Default = training')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing recordings (required).')
    parser.add_argument('--name', type=str, default=None, help='Class name (required).')
    parser.add_argument('--offset', type=float, default=2.5, help=f'Get a spectrogram at every <this many> seconds. Default = 2.5.')
    parser.add_argument('--max_all', type=int, default=None, help=f'Total maximum number of spectrograms to insert, if specified (no default).')
    parser.add_argument('--max_per', type=int, default=None, help=f'Maximum number of spectrograms to insert per recording, if specified (no default).')

    args = parser.parse_args()
    if args.dir is None:
        print("Error: --dir argument is required (directory containing recordings).")
        quit()
    else:
        audio_path = args.dir

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

    ExtractAll(audio_path, args.db, class_name, class_code, args.offset, args.max_all, args.max_per).run()

    elapsed = time.time() - run_start_time
    print(f'elapsed seconds = {elapsed:.1f}')
