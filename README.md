## Introduction
BRITE-Kit (Bioacoustic Recognizer Technology Kit) is a toolkit for building your own recognizer. It is licensed under the terms of the MIT license. BRITE-Kit is essentially a stripped down version of [HawkEars](https://github.com/jhuus/BRITE-Kit), without the bird-specific features, and with much of the complexity removed.

BRITE-Kit models are trained on spectrograms. By default, spectrograms represent 5-second segments, but you have full control over segment length and parameters such as the minimum and maximum frequency. A selection of leading model types and sizes is included, along with scripts to perform functions such as:

- Download recordings from iNaturalist or Google Audioset
- Generate spectrogram images for whole recordings or segments
- Extract all or selected spectrograms into a training database
- Perform operations on the training database, such as reporting, deleting a class, plotting spectrograms, or deleting selected spectrograms
- Train a model
- Run inference

## Installation

To install BRITE-Kit on Linux or Windows:

1.	Install [Python 3](https://www.python.org/downloads/), if you do not already have it installed.
2.	Install git.
3.  Type

```
 git clone https://github.com/jhuus/BRITE-Kit
 cd BRITE-Kit
```

4.	Install required Python libraries:

```
pip install -r requirements.txt
```

5. Install SQLite. On Windows, follow [these instructions](https://www.sqlitetutorial.net/download-install-sqlite/). On Linux, type:

```
sudo apt install sqlite3
```

6. If you have a [CUDA-compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus), such as a Geforce RTX, you can gain a major performance improvement by installing [CUDA](https://docs.nvidia.com/cuda/).

## Developing Your Own Recognizer
### Overview
Developing a recognizer with BRITE-Kit involves iterating on the following steps.

1. Collect training data
2. Create a training database
3. Train one or more models
4. Test your models

### Collecting Training Data
#### Introduction
You can use your own recordings, or recordings from public source such as:

- [Google Audioset](https://research.google.com/audioset/)
- [iNaturalist](https://www.inaturalist.org/)
- [Freefield1010](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/qmul_freefield1010.html)

In addition, BRITE-Kit includes a small selection of noise recordings under data/recordings/Noise. It is very important to include a diverse set of background noise spectrograms. A class called Noise is required, and should include only wind and rain sounds. Other non-target sounds such as speech or traffic can be included in separate classes, or grouped together into a class called Other. The class name Noise is required, but other class names can be anything you choose.
#### Google Audioset
Google Audioset is simply metadata for the audio in a large set of Youtube recordings. The metadata is included in the data/audioset directory. There are 527 defined classes, listed in class_list.csv. For example, the first five classes are:

1. Speech
2. Male speech, man speaking
3. Female speech, woman speaking
4. Child speech, kid speaking
5. Conversation

The file unbalanced_train_segments.csv tags 10-second Youtube clips with one or more of these classes, and the script tools/audioset.py lets you download corresponding clips. Most clips have multiple tags, and in many cases these include undesirable classes. For example, when downloading speech, it is usually fine to include any of the five classes listed above, but not music, or barking dogs etc. To support this need, the file class_inclusion.csv (specific to BRITE-Kit) defines acceptable secondary tags. For Speech, it says that the other four tags above are acceptable, but only those.

Even if you restrict the downloads based on class_inclusion.csv, you will find some clips include other sound classes, or are problematic in other ways. To address this, BRITE-Kit provides a data/audioset/curated directory with a list of curated segments for a number of classes. For example, 300 10-second wind clips are listed in wind.csv.

- To download curated clips, use the script tools/download_curated.csv
- To see which secondary classes are associated with a class, use tools/audioset.py with the -r flag
- To download an uncurated class, use tools/audioset.py without the -r flag
- To create or update a list of curated segments, use tools/curate.py

#### iNaturalist
iNaturalist is another good source of recordings, especially for birds, amphibians, insects and mammals. The script tools/inat.py lets you download recordings from iNaturalist. It is strongly recommended to use the scientific name rather than the common name when possible though. For example, if you request Leopard Frog, the downloaded recordings will include sounds of the Rio Grande Leopard Frog, so it is much better to request Lithobates pipiens.

### Plotting Spectrograms from Recordings
The tools/plot_recordings.py script generates spectrogram images (jpeg files) for each recording in a specified directory. If you specify --all, each spectrogram will represent an entire recording. Otherwise, by default each spectrogram will represent a 5-second non-overlapping segment. You can override these parameters with the --seconds and --overlap arguments, and you can change the defaults by editing core/base_config.py.

### Populating a Training Database
Training data is stored in a SQLite database, in data/training.db by default. There are two ways to add data to the database:

1. Use tools/extract_all.py to insert spectrograms for every segment of every recording in a directory.
2. Use tools/extract_by_image.py to insert spectrograms corresponding to spectrogram images in a directory.

For example, to initialize a Noise class with all the segments in the provided noise recordings, type this in the tools directory:

```
python extract_all.py --code Noise --name Noise --dir ../data/recordings/Noise
```

If the database does not exist, this will create it and create a class called Noise with 5-second spectrograms taken at 2.5-second increments for all the specified recordings. The same approach can be used for the curated Google Audioset classes. Note that every class must be given a name and a code.

If you download recordings from iNaturalist though, e.g. for Northern Leopard Frog, you probably shoudn't just insert them all into the database. Instead, you can use tools/plot_recordings.py to plot the segments, then delete the undesirable image files and use tools/extract_by_image.py to insert training records corresponding to the remaining images.

There are a number of other scripts in the tools directory that help with managing the training database:

1. db_report.py generates two CSV files that describe the database contents
2. plot_from_db.py lets you plot the spectrograms for a class in the database.
3. del_class.py lets you delete a class.
4. del_spec.py lets you delete the spectrograms corresponding to images in a directory. First use plot_from_db.py then delete the images you want to keep (or copy the ones you want to delete to another directory), and use del_spec.py to delete the ones you don't want to keep.
5. del_empty_recordings.py lets you delete any recordings that have no associated spectrograms. This can happen if you get aggressive with del_spec.py, and you can check for it by running db_report.py.

### Training
The first step in training is to run tools/pickle_db.py. It extracts the data from the training database into a binary file (data/training.pkl) that is used by the training process. Typically you will run training many times between database updates, and using a binary input file is faster than reading the database directly.

Next you need to define your training parameters by editing core/configs.py. You can override any of the parameters defined in the training section of core/base_config.py, but the main ones are:

1. Model type
2. Number of epochs
3. Proportion of training data reserved for validation

For model type, it is usually best to use "custom_x_y", where x is "hgnet", "efficientnet" or "vovnet" and y is an integer. The model directory contains definitions for these three models, and lists the corresponding model sizes. For example, "custom_vovnet_1" gives a model with about 50K parameters, while "custom_vovnet_8" has about 3.1M parameters. It's usually best to use the smallest model that produces good results. You don't have to use custom models though. As an alternative, you can use any model name supported by [timm](https://github.com/huggingface/pytorch-image-models), e.g. "tf_efficientnet_b0".

For the first training run, start with a small number of epochs, e.g. 5, so you can do some basic validation before trying more. Specifying val_portion=.1 uses 90% of the data for training and 10% for validation, which lets you see how well the model is working on training data.

The output of training is stored under the logs directory. For instance, the first run will be under logs/fold-0/version_0. Checkpoints are stored under the checkpoints directory. To view a graph of the training loss over time, type:

```
tensorboard --logdir logs/fold-0/version_0
```

This will print a URL that you can paste into your browser to see the reports.

### Using Your Trained Models
To use a trained model, copy the checkpoint file into data/ckpt, then run analyze.py to run inference on some recordings. If there are multiple checkpoint files in data/ckpt, they will be treated as an ensemble. That is, the average of all their predictions will be used. In this case it's very important that all checkpoints represent the same classes though!

The file data/ignore.txt lists classes that are excluded from inference output. By default, only Noise is excluded, but you can edit ignore.txt to add other classes such as Speech.



