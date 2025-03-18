# Delete all spectrograms and recordings for a class from a database.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import util

parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, default='training', help='Database name. Default = training')
parser.add_argument('--name', type=str, default='', help='Class name')
args = parser.parse_args()

db_name = args.db
class_name = args.name

database = util.get_database(db_name)
database.delete_spectrogram_by_subcat_name(class_name)
database.delete_recording_by_subcat_name(class_name)
database.delete_subcategory('Name', class_name)
