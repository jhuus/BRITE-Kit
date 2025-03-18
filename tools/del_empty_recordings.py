# Delete any recordings that have no spectrograms.

import argparse
import logging
import os
import inspect
import sys
from pathlib import Path
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import plot
from core import util

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, default='training', help='Database name. Default = training')
parser.add_argument('--name', type=str, default='', help='Class name')
args = parser.parse_args()

db_name = args.db
class_name = args.name

db = util.get_database(db_name)
results = db.get_recording_by_subcat_name(class_name)
for r in results:
    count = db.get_spectrogram_count_by_recid(r.id)
    if count == 0:
        db.delete_recording(value=r.id)
