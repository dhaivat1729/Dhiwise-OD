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
                        isfile,
                        build_detectron2_config)

from DWODLib.dataset import get_dataset_fiftyone, split_fiftyone_dataset, convert_fo_to_detectron2
from DWODLib.engine import Trainer

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
make_dir(config['outputDir'])

## save python command to reproduce results
save_command(os.path.join(config['outputDir'], 'command.txt'))

## build python command to reproduce results
build_command('tools/train.py', config, os.path.join(config['outputDir'], 'defaultCommand.sh'))

## Build dataset in fiftyone format
dataset, class2Id = get_dataset_fiftyone(config)
config['num_classes'] = len(class2Id)
save_json(class2Id, os.path.join(config['outputDir'], 'class2Id.json')) ## save mapping from class to id

## Split dataset
dataset = split_fiftyone_dataset(dataset, config)
convert_fo_to_detectron2(dataset, class2Id)

## Build model
det2Config = build_detectron2_config(config)

## save our config
save_json(config, os.path.join(config['outputDir'], 'experiment_config.json'))

## Trainer object
trainer = Trainer(det2Config)

## train the model
trainer.dt.resume_or_load(resume=False)
trainer.dt.train()