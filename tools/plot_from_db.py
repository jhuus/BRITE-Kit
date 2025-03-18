# Generate an image file for every spectrogram for the specified class/database.

import argparse
import inspect
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import database
from core import plot
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db', type=str, default=cfg.train.training_db, help='Database name.')
parser.add_argument('-e', '--exp', type=float, default=.8, help='Raise spectrograms to this exponent.')
parser.add_argument('-n', '--max', type=int, default=0, help='If > 0, stop after this many images. Default = 0.')
parser.add_argument('-s', '--name', type=str, default='', help='Class name.')
parser.add_argument('-o', '--out', type=str, default='', help='Output directory.')

args = parser.parse_args()

db_path = f"../data/{args.db}.db"
exponent = args.exp
num_to_plot = args.max
species_name = args.name
out_dir = args.out

if not os.path.exists(out_dir):
    print(f'creating directory {out_dir}')
    os.makedirs(out_dir)

db = database.Database(db_path)
results = db.get_spectrogram_by_subcat_name(species_name, include_ignored=True)
num_plotted = 0
for r in results:
    base, ext = os.path.splitext(r.filename)
    spec_path = f'{out_dir}/{base}-{r.offset:.2f}.jpeg'

    if not os.path.exists(spec_path):
        print(f"Processing {spec_path}")
        spec = util.expand_spectrogram(r.value)
        if np.min(spec) < 0 or np.max(spec) != 1:
            print(f"    min={np.min(spec)}, max={np.max(spec)}")

        num_plotted += 1
        plot.plot_spec(spec ** exponent, spec_path)

    if num_to_plot > 0 and num_plotted == num_to_plot:
        break

