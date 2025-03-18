# Output a report summarizing database contents.

import argparse
import inspect
import os
import sys
from pathlib import Path
import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database

parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, default='training', help='Database name. Default = training')
parser.add_argument('-o', type=str, default=None, help='Output directory')
args = parser.parse_args()

db_name = args.db
output_dir = args.o
if output_dir is None:
    print(f"Error: -o argument (output directory) is required")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

db_path = str(Path("..") / Path("data") / Path(args.db + ".db"))
db = database.Database(db_path)

# create summary report (spectrogram count per class)
classes = db.get_subcategory()
recordings_per_class = {}
names, codes, recording_count, spec_count = [], [], [], []
for c in sorted(classes, key=lambda obj: obj.name):
    names.append(c.name)
    codes.append(c.code)
    results = db.get_recording_by_subcat_name(c.name)
    recordings_per_class[c.name] = results
    recording_count.append(len(results))
    spec_count.append(db.get_spectrogram_count(c.name))

df = pd.DataFrame()
df['name'] = names
df['code'] = codes
df['recordings'] = recording_count
df['spectrograms'] = spec_count
df.to_csv(os.path.join(output_dir, "class_summary.csv"), index=False)

# create details report (spectrogram count per recording per class)
names, codes, recordings, spec_count = [], [], [], []
for c in sorted(classes, key=lambda obj: obj.name):
    for r in sorted(recordings_per_class[c.name], key=lambda obj: obj.filename):
        names.append(c.name)
        codes.append(c.code)
        recordings.append(r.filename)
        spec_count.append(db.get_spectrogram_count_by_recid(r.id))

df = pd.DataFrame()
df['name'] = names
df['code'] = codes
df['recording'] = recordings
df['spectrograms'] = spec_count
df.to_csv(os.path.join(output_dir, "class_details.csv"), index=False)
