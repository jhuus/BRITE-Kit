# Each configuration is a dataclass that sets the members it needs,
# as in the examples below. To add a config, add a new dataclass here
# and then add it to the configs dictionary below so it has a name.
# This approach supports typeahead and error-checking, which are very useful.

from dataclasses import dataclass
from core import base_config

cfg : base_config.BaseConfig = base_config.BaseConfig()

# Simple training example
@dataclass
class Train(base_config.BaseConfig):
    def __init__(self):
        self.train.model_name = "custom_vovnet_2"
        self.train.num_epochs = 8
        self.train.val_portion = .1

# map names to configurations
configs = {"base": base_config.BaseConfig,
           "train": Train,
          }

# set a configuration based on the name
def set_config(name):
    if name in configs:
        cfg = configs[name]()
    else:
        raise Exception(f"Configuration '{name}' not defined")
