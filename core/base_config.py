# Base configuration. Specific configurations only have to specify the parameters they're changing.

from enum import IntEnum
from dataclasses import dataclass

@dataclass
class Audio:
    segment_len = 5         # spectrogram duration in seconds
    spec_height = 128       # spectrogram height
    spec_width = 256        # spectrogram width
    sampling_rate = 32000
    win_length = 2048
    min_audio_freq = 0
    max_audio_freq = 8000
    mel_scale = True
    power = 1.0
    spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

@dataclass
class Training:
    multi_label = True
    deterministic = False
    seed = None
    fast_optimizer = False  # slow optimizer is more accurate but takes 25-30% longer to train
    learning_rate = .0025   # base learning rate
    batch_size = 16
    model_name = "custom_hgnet_1"
    use_class_weights = True
    num_workers = 2
    dropout = None          # various dropout parameters are passed to model only if not None
    drop_rate = None
    drop_path_rate = None
    proj_drop_rate = None
    num_epochs = 10
    LR_epochs = None        # default = num_epochs, higher values reduce effective learning rate decay
    save_last_n = 3         # save checkpoints for this many last epochs
    label_smoothing = 0
    training_db = "training" # name of training database
    num_folds = 1           # for k-fold cross-validation
    val_portion = 0         # used only if num_folds = 1
    model_print_path = "model.txt" # path of text file to print the model (TODO: put in current log directory)

    # data augmentation (see core/dataset.py to understand these parameters)
    augmentation = True
    prob_simple_merge = 0.35
    prob_real_noise = 0.3
    prob_speckle = .1
    prob_fade1 = .2
    prob_fade2 = 1
    prob_shift = 1
    max_shift = 6
    min_fade1 = .1
    max_fade1 = .8
    min_fade2 = .1
    max_fade2 = 1
    speckle_variance = .012

@dataclass
class Inference:
    num_threads = 3              # multiple threads improves performance but uses more GPU memory
    spec_overlap_seconds = 0     # number of seconds overlap for adjacent spectrograms
    min_score = 0.80             # only generate labels when score is at least this
    audio_exponent = .85         # power parameter for mel spectrograms during inference
    block_size = 100             # do this many spectrograms at a time to avoid running out of GPU memory
    seed = 99                    # reduce non-determinism during inference

@dataclass
class Miscellaneous:
    main_ckpt_folder = "data/ckpt"      # use an ensemble of all checkpoints in this folder for inference
    ignore_file = "data/ignore.txt"     # classes listed in this file are ignored in analysis
    train_pickle = "data/training.pkl"
    test_pickle = None

@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
