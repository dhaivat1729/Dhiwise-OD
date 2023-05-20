
import fiftyone as fo
from PIL import Image
import os
from tqdm import tqdm
from DWODLib.config import defaultCfg
from DWODLib.utils import (isfile, 
                                load_json, 
                                save_json,)

baseDir = defaultCfg['dataDir']
annotationFile = os.path.join(baseDir, defaultCfg['annotationFile'])
imageDir = os.path.join(baseDir, defaultCfg['imageDir'])

## loading annotations
data = load_json(annotationFile) ## this is a list of images

samples = []

## let's put data in fiftyone format
for image in tqdm(data):
    fileName = os.path.join(imageDir, image['screenName'])

    if isfile(fileName):

        ## create fiftyone
        sample = fo.Sample(filepath=fileName)

        ## get image class
        sample['class'] = image['class']

        ## load file anmd get size
        img = Image.open(fileName)
        width, height = img.size

        ## storing sample size
        sample['size'] = {'width':width, 'height':height}
        
        ## detections
        detections = []

        ## normalize bounding boxes
        for bbox in image['children']:

            ## getting normalized bounding box
            bbox['x'] = bbox['x'] / width
            bbox['y'] = bbox['y'] / height
            bbox['width'] = bbox['width'] / width
            bbox['height'] = bbox['height'] / height
            

            ## building fiftyone detection
            detection = fo.Detection(label=bbox['type'], bounding_box=[bbox['x'], bbox['y'], bbox['width'], bbox['height']])
            detections.append(detection)

        ## add bounding boxes to fiftyone
        sample['ground_truth'] = fo.Detections(detections=detections)
        samples.append(sample)

dataset = fo.Dataset("Dhiwise object detection")
dataset.add_samples(samples)
session = fo.launch_app(dataset)
session.wait()