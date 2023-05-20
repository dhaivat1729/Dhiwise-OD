## build detectron2 config from our config file

from detectron2.config import get_cfg
from detectron2 import model_zoo

def build_custom_config(args_dict):
    """
        Build detectron2 config from our config file
    """
    det2_cfgFile = args_dict['detectorName']
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(det2_cfgFile))
    cfg.DATASETS.TRAIN = ("fiftyone_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(det2_cfgFile)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args_dict['batchSize']  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = args_dict['learningRate']  # pick a good LR
    cfg.SOLVER.MAX_ITER = args_dict['numSteps']    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (Vehicle registration plate). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    
    cfg.freeze()
    return cfg