
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
    'save_after_steps':100, ## save model after every 100 steps
    'bbox_loss' : 'smooth_l1', ## smooth_l1, giou, diou, ciou
    'imageMinMaxSizes' : [800, 1333], ## min and max sizes of shortest side of image
    'cropEnabled': False, ## crop images during training
    'cropSize': [0.9, 0.9] ## crop size during training
}

defaultCfg = ConfigDict(defaultCfg)