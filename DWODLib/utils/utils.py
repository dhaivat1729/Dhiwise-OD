"""
    A utility file with different utility functions. This will help in avoiding code cluttering. 
"""

from ast import literal_eval
import json
from pathlib import Path
from collections import OrderedDict
import datetime
from copy import deepcopy
import sys

###################################################################################################################################################


def set_seed(seed):
    """
        Set seed for reproducibility
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

## save dictionary as json
def save_json(data, filename, indent=4, sort_keys=True):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)

## load json file
def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return data

## save python class object as json
def save_obj_as_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj.__dict__, f, indent=4, sort_keys=True)

## load json file as python class object
def load_json_as_obj(obj, filename):
    with open(filename, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    obj.__dict__ = data
    return obj

## write text to file
def write_text(text, filename):
    Path(filename).write_text(text)

## save python command from arguments
def save_command(filename):
    command = 'python3 ' + ' '.join(sys.argv)
    write_text(command, filename)

## build python command to reproduce results
def build_command(script, config, filepath):
    command = f"python3 {script} --options " + ' '.join([f'"--{key}" "{value}"' for key, value in config.items()]) + '\n'
    write_text(command, filepath)

## read text
def read_text(filename, delimiter=None):
    if delimiter is None:
        return Path(filename).read_text()
    else:
        return Path(filename).read_text(encoding='utf-8').split(delimiter)
    
## make dir
def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

## check if dir exists
def isdir(path):
    return Path(path).is_dir()

## check if file exists
def isfile(path):
    return Path(path).is_file()

## get parent path as string
def get_parent_path(path):
    return str(Path(path).parent)

## get current time as string
def get_current_time():    
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")