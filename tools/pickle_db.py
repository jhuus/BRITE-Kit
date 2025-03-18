# Create and pickle a dict containing two dataframes: one with class (species) info and one with spectrograms.
# Training can then load this efficiently, and it reduces changes in the main code when we change the training
# data source and format.

import argparse
import inspect
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import database
from core import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
output_path = str(Path("..") / Path("data") / Path("training.pkl"))

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db', type=str, default=cfg.train.training_db, help=f'Database name. Default = "{cfg.train.training_db}".')
parser.add_argument('-o', '--out', type=str, default=f"{output_path}", help=f'Output path. Default = "{output_path}".')

args = parser.parse_args()
output_path = args.out
db = util.get_database(args.db)

# get class names and codes
classes = db.get_subcategory()
class_names, class_codes = [], []
for c in sorted(classes, key=lambda obj: obj.name):
    class_names.append(c.name)
    class_codes.append(c.code)

# map recording IDs to file names
recording_dict = {}
results = db.get_recording()
for r in results:
    recording_dict[r.id] = r.filename

# get num spectrograms total and per class
logging.info('Counting spectrograms per class')
total_specs = 0
num_spectrograms = []
for i in range(len(class_names)):
    count = db.get_spectrogram_count(class_names[i])
    logging.info(f'# spectrograms for {class_names[i]} = {count}')
    num_spectrograms.append(count)
    total_specs += count

logging.info(f'Total # spectrograms: {total_specs}')

# get spectrograms
specs_per_class = {}
for i, class_name in enumerate(class_names):
    specs_per_class[class_name] = []
    results = db.get_recording_by_subcat_name(class_name)
    for r in results:
        results2 = db.get_spectrogram('RecordingID', r.id)
        for r2 in results2:
            rec_name = recording_dict[r2.recording_id]
            specs_per_class[class_name].append([rec_name, r2.offset, r2.value])

spec = [0 for i in range(total_specs)]
rec_name = ['' for i in range(total_specs)]
offset = [0 for i in range(total_specs)]
spec_index = [i for i in range(total_specs)]
class_index = np.zeros((total_specs, ), dtype=np.int32)
idx = 0
for i, class_name in enumerate(class_names):
    permutation = np.random.permutation(len(specs_per_class[class_name]))
    for j in range(num_spectrograms[i]):
        _rec_name, _offset, _spec = specs_per_class[class_name][permutation[j]]
        offset[idx] = _offset
        spec[idx] = _spec
        rec_name[idx] = _rec_name
        class_index[idx] = i
        idx += 1

# create dataframes, then pickle a dict containing them
spec_df = pd.DataFrame({'spec': spec, 'spec_index': spec_index, 'rec_name': rec_name, 'offset': offset, 'class_index': class_index})
class_df = pd.DataFrame({'name': class_names, 'code': class_codes})
save_dict = {'spec': spec_df, 'class': class_df}

pickle_file = open(output_path, 'wb')
pickle.dump(save_dict, pickle_file)
logging.info(f"Database exported to {output_path}")
