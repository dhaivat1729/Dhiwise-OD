"""
   Inference on image directory using a trained model 
"""


### train object detection model with detectron2 and fiftyone

## imports
import os
from tqdm import tqdm
from DWODLib.config import defaultCfg as config
from DWODLib.utils import (get_args, 
						load_json,
						build_detectron2_config,
						merge_default_config,
						make_dir,
						save_json)

from DWODLib.dataset import convert_detectron2_to_fo, convert_fiftyone_to_dhiwise
from DWODLib.engine import Predictor
import cv2
import glob
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import fiftyone as fo

## usage: python3 tools/inference.py --options --inputImagesDir '/path/to/input/images' --modelPathDirectory '/path/to/model/directory' --outputImageDir '/path/to/output/images' --visualize "True" --outputResultsFile "/path/to/output/results/file.json"

## get arguments
args = get_args()

## model directory has to be there
assert args['modelPathDirectory'] is not None, "Model directory has to be specified for inference!"

## output image directory has to be there
assert args['outputImageDir'] is not None, "Output image directory has to be specified for storing images!"

## input image directory has to be there
assert args['inputImagesDir'] is not None, "Input image directory has to be specified for inference!"

## make output directory
make_dir(args['outputImageDir'])

## override default config with new arguments
expconfig = load_json(os.path.join(args['modelPathDirectory'], 'experiment_config.json'))
config = merge_default_config(config, expconfig)

## Build dataset in fiftyone format
class2Id = load_json(os.path.join(config['outputDir'], 'class2Id.json')) ## save mapping from class to id

## detectron2 model
det2Config = build_detectron2_config(config)
det2Config.MODEL.WEIGHTS = os.path.join(config['outputDir'], 'model_final.pth') ## model weights
det2Config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args['testScoreThresh']  if 'testScoreThresh' in args else 0.7 ## set threshold for this model

## Predictor object
predictor = Predictor(det2Config)

## list of images
images = glob.glob(os.path.join(args['inputImagesDir'], '*.jpg'))

## ensure that there are images in the input directory
assert len(images) > 0, "No images found in the input directory!"

## set classes metadata
MetadataCatalog.get(det2Config.DATASETS.TRAIN[0]).set(thing_classes=list(class2Id.keys()))

## output images
outputImages = []
samples = []
## run inference on each image
for image in tqdm(images):

	sample = fo.Sample(filepath=image)

	imageName = os.path.basename(image) ## name of the image
	img = cv2.imread(image) ## read image

	## output image path
	outputImagePath = os.path.join(args['outputImageDir'], imageName)

	## run inference
	outputs = predictor.dp(img)

	## visualize
	v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(det2Config.DATASETS.TRAIN[0]), scale=1.0)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

	image_h, image_w, _ = img.shape
	sample['detections'] = convert_detectron2_to_fo(outputs, image_w, image_h, det2Config.DATASETS.TRAIN[0])
	sample['filepath'] = outputImagePath
	## iamge height and width
	sample['height'] = image_h
	sample['width'] = image_w
	samples.append(sample)

	## save image
	cv2.imwrite(outputImagePath, v.get_image()[:, :, ::-1])

## fiftyone dataset
dataset = fo.Dataset("Dhiwise object detection")
dataset.add_samples(samples)

## convert fiftyone to dhiwise format
dhiwiseFormat = convert_fiftyone_to_dhiwise(dataset)

## save dhiwise format
if args['outputResultsFile'] is not None:
	save_json(dhiwiseFormat, args['outputResultsFile'])

if args['visualize']:
	## launch fiftyone app
	session = fo.launch_app(dataset)
	session.wait()
