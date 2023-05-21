"""
   Inference on a dataset using a trained model 
"""


### train object detection model with detectron2 and fiftyone

## imports
import os
from tqdm import tqdm
from DWODLib.config import defaultCfg as config
from DWODLib.utils import (get_args, 
                        load_json,
                        build_detectron2_config)

from DWODLib.dataset import get_dataset_fiftyone, split_fiftyone_dataset, convert_fo_to_detectron2, get_fiftyone_dicts, convert_detectron2_to_fo
from DWODLib.engine import Predictor
import cv2
## get detectron2 metadata
from detectron2.data import MetadataCatalog, DatasetCatalog
import fiftyone as fo

## get arguments
args = get_args()

## output directory has to be there
assert args['outputDir'] is not None, "Output directory has to be specified for inference!"

## override default config with new arguments
config = load_json(os.path.join(args['outputDir'], 'experiment_config.json'))
## Build dataset in fiftyone format
class2Id = load_json(os.path.join(config['outputDir'], 'class2Id.json')) ## save mapping from class to id
dataset, _ = get_dataset_fiftyone(config)

## Split dataset
dataset = split_fiftyone_dataset(dataset, config)
convert_fo_to_detectron2(dataset, class2Id)

## Build model
det2Config = build_detectron2_config(config)
det2Config.MODEL.WEIGHTS = os.path.join(config['outputDir'], 'model_final.pth') ## model weights
det2Config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args['testScoreThresh']  if 'testScoreThresh' in args else 0.7 ## set threshold for this model

## get dataset name so we can extract corresponding class label information
trainDatasetName = det2Config.DATASETS.TRAIN[0]
testDataset = trainDatasetName.replace("train", "val")

## Predictor object
predictor = Predictor(det2Config)

## run evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator(testDataset, output_dir=config['outputDir'])
# val_loader = build_detection_test_loader(det2Config, testDataset)
# print(inference_on_dataset(predictor.dp.model, val_loader, evaluator))

## Run inference
val_view = dataset.match_tags("val")
dataset_dicts = get_fiftyone_dicts(val_view, class2Id)
predictions = {}
for d in tqdm(dataset_dicts):
    img_w = d["width"]
    img_h = d["height"]
    img = cv2.imread(d["file_name"])
    outputs = predictor.dp(img)
    detections = convert_detectron2_to_fo(outputs, img_w, img_h, testDataset)
    predictions[d["image_id"]] = detections

dataset.set_values("predictions", predictions, key_field="id")

## launch fiftyone app
session = fo.launch_app(dataset, remote=True)
session.wait()
