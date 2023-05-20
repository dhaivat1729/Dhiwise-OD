"""
    Skeleton for training object detection model with detectron2 and fiftyone
    1. Load dataset from annotations in fiftyone format, split dataset
    2. Build model
    3. Convert dataset to detectron2 format
    4. Train model
    5. Evaluate model
    6. Save model
"""


### train object detection model with detectron2 and fiftyone

## imports
import os
from DWODLib.config import defaultCfg as config
from DWODLib.utils import (get_args, 
                        make_dir, 
                        save_json, 
                        load_json,
                        save_command, 
                        build_command,
                        isdir,
                        isfile,)

from DWODLib.dataset import get_dataset_fiftyone, split_fiftyone_dataset

## get arguments
args = get_args()

## output directory has to be there
assert args['outputDir'] is not None, "Output directory has to be specified!"

if isdir(args['outputDir']):
    ## if output directory already exists, we can't overwrite it
    raise FileExistsError(f"Output directory {args['outputDir']} already exists! Please rerun with a different output directory.")
    exit()

## override default config with new arguments
config.update(args)

## make output directory
make_dir(config.OUTPUT_DIR)

## save python command to reproduce results
save_command(os.path.join(config.OUTPUT_DIR, 'command.txt'))

## build python command to reproduce results
build_command('tools/train.py', config, os.path.join(config.OUTPUT_DIR, 'defaultCommand.sh'))

## Build dataset in fiftyone format
dataset = get_dataset_fiftyone(config)

## Split dataset
dataset = split_fiftyone_dataset(dataset, config)