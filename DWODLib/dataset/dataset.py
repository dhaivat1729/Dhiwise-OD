## utilities relatex to dataset loading

import fiftyone as fo
from PIL import Image
import os
from tqdm import tqdm
from DWODLib.config import ConfigDict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from DWODLib.utils import (isfile, 
                        load_json,)

import fiftyone.utils.random as four

def convert_json_from_list_to_dict(jsonFile):
    """
        Current json file is of type list, 
        we convert it into a dictionary where key is image name.
    """    
    data = load_json(jsonFile)

    ## new dictionary
    converted_json = {}

    for image in tqdm(data):
        converted_json[image['screenName']] = image

    return converted_json

def build_fiftyone_from_converted_json(convertedJsons, filetypes, imageDir):
    """
        This takes list of converted jsons (which is output of convert_json_from_list_to_dict() function)
    """

    ## samples
    samples = []

    ## assert that all convertedJsons have same keys
    assert len(convertedJsons) > 0, "No converted jsons found!"

    ## build set of keys for every json file and take intersection of all keys
    keys_sets = [set(convertedJson.keys()) for convertedJson in convertedJsons]
    common_keys = set.intersection(*keys_sets)

    ## iterate through all keys
    for key in tqdm(common_keys):
            
        ## path of the image
        filePath = os.path.join(imageDir, key)

        ## If image is a file, then only we register its annotations
        if isfile(filePath):
            
            ## create fiftyone  sample for this image
            sample = fo.Sample(filepath=filePath)

            ## load file anmd get size
            img = Image.open(filePath)
            width, height = img.size

            ## storing sample size
            sample['size'] = {'width':width, 'height':height}

            ## iterate through all convertedJsons
            for convertedJson, filetype in zip(convertedJsons, filetypes):

                ## get image class
                sample['class'] = convertedJson[key]['class']
                
                detections = []
                ## normalize bounding boxes
                for bbox in convertedJson[key]['children']:

                    ## getting normalized bounding box
                    bbox['x'] = bbox['x'] / width
                    bbox['y'] = bbox['y'] / height
                    bbox['width'] = bbox['width'] / width
                    bbox['height'] = bbox['height'] / height

                    ## building fiftyone detection
                    detection = fo.Detection(label=bbox['type'], bounding_box=[bbox['x'], bbox['y'], bbox['width'], bbox['height']])
                    detections.append(detection)
                
                sample[filetype] = fo.Detections(detections=detections)
            samples.append(sample)

    dataset = fo.Dataset("Dhiwise object detection")
    dataset.add_samples(samples)
    return dataset

def get_dataset_fiftyone(config):
    ## base directory and annotation file
    baseDir = config['dataDir']
    annotationFile = os.path.join(baseDir, config['annotationFile'])
    imageDir = os.path.join(baseDir, config['imageDir'])
    
    ## loading annotations  
    data = load_json(annotationFile) ## this is a list of images
    samples = []
    class2Id = set() ## set
    
    ## let's put data in fiftyone format
    for image in tqdm(data):

        ## path of the image
        filePath = os.path.join(imageDir, image['screenName'])

        ## If image is a file, then only we register its annotations
        if isfile(filePath):
                
            ## create fiftyone  sample for this image
            sample = fo.Sample(filepath=filePath)

            ## get image class
            sample['class'] = image['class']

            ## load file anmd get size
            img = Image.open(filePath)
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
                
                ## if class is not in class2Id, then add it
                class2Id.add(bbox['type'])

                ## building fiftyone detection
                detection = fo.Detection(label=bbox['type'], bounding_box=[bbox['x'], bbox['y'], bbox['width'], bbox['height']])
                detections.append(detection)

            ## add bounding boxes to fiftyone
            sample['ground_truth'] = fo.Detections(detections=detections)
            samples.append(sample)

    # datasetName = 'Dhiwise object detection' if time == 'train' else 'Dhiwise object detection validation'
    dataset = fo.Dataset("Dhiwise object detection")
    dataset.add_samples(samples)

    ## class2ID must be in sorted order
    class2Id = sorted(list(class2Id))
    class2Id = {class2Id[i]:i for i in range(len(class2Id))} ## this ensures reproducibility when we go in any order

    return dataset, class2Id
    

def split_fiftyone_dataset(dataset : fo.Dataset, config : ConfigDict):
    
    ## Divide dataset into train and validation
    trainPartition, valPartition = config['trainPartition'], config['valPartition']
    
    four.random_split(dataset, {"train": trainPartition, "val": valPartition}, seed=config['seed']) 
    return dataset


def get_fiftyone_dicts(samples, class2Id):
    """
        Convert fiftyone samples to detectron2 format
    """
    
    samples.compute_metadata()

    ## let's iterate through samples
    dataset_dicts = []
    for sample in samples:
        
        ## getting metadata for detectron2
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        record["height"] = height
        record["width"] = width
      
        objs = []
        ## iterating through all object and registering them 
        for det in sample.ground_truth.detections:
            tlx, tly, w, h = det.bounding_box
            ## fiftyone has detections between [0,1]
            bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
            ## Object details
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": class2Id[det.label],
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def convert_fo_to_detectron2(dataset, class2Id):
    for d in ["train", "val"]:
        view = dataset.match_tags(d)
        DatasetCatalog.register("fiftyone_" + d, lambda view=view: get_fiftyone_dicts(view, class2Id))
        MetadataCatalog.get("fiftyone_" + d).set(thing_classes=[key for key in class2Id.keys()])

def convert_detectron2_to_fo(outputs, image_w, image_h, datasetName):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    class_names = MetadataCatalog.get(datasetName).thing_classes
    pred_class_names = list(map(lambda x: class_names[x], instances.pred_classes))
    
    for pred_box, score, c in zip(instances.pred_boxes, instances.scores, pred_class_names):
        x1, y1, x2, y2 = pred_box
        bbox = [float(x1)/image_w, float(y1)/image_h, float(x2-x1)/image_w, float(y2-y1)/image_h]
        detection = fo.Detection(label=c, confidence=float(score), bounding_box=bbox)
        detections.append(detection)

    return fo.Detections(detections=detections)


def convert_fiftyone_to_dhiwise(dataset):
    """
        Convert fiftyone dataset to dhiwise format
    """
    ## get all samples
    samples = dataset.to_dict()['samples']

    ## new dictionary
    dhiwise_format = []

    for sample in tqdm(samples):
        converted_sample = {}
        converted_sample['screenName'] = os.path.basename(sample['filepath'])
        converted_sample['class'] = sample['class'] if 'class' in sample else 'Mobile'
        converted_sample['children'] = []

        ## get image size
        width = sample['width']
        height = sample['height']

        ## iterate through all detections
        for detection in sample['detections']['detections']:
            bbox = detection['bounding_box']
            converted_bbox = {}
            converted_bbox['x'] = bbox[0] * width
            converted_bbox['y'] = bbox[1] * height
            converted_bbox['width'] = bbox[2] * width
            converted_bbox['height'] = bbox[3] * height
            converted_bbox['type'] = detection['label']
            converted_sample['children'].append(converted_bbox)

        dhiwise_format.append(converted_sample)

    return dhiwise_format