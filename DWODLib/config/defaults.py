
from .config import ConfigDict

defaultCfg = {
    'detectorName':'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', ## detectron2 model name
    'dataDir':'/Users/dhaivat1729/Downloads/data_OD/',
    'annotationFile':'annotation.json',
    'imageDir':'images',
    'outputDir':None, ## To be provided at train time
    'batchSize':2,
    'learningRate':0.00025,
    'numSteps':300,
    'numClasses':5, ## this will have to be changed
    'seed':42, ## for reproducibility
    'trainPartition':0.8, ## 80% of data for training
    'valPartition':0.2, ## 20% of data for validation
    'freeze_RPN':False, ## freeze RPN weights
    'freeze_backbone':False, ## freeze backbone weights
}

defaultCfg = ConfigDict(defaultCfg)