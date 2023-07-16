## build detectron2 config from our config file

from detectron2.config import get_cfg
from detectron2 import model_zoo

## borrow other variables from default config 
def merge_default_config(config, expconfig):
    for key, value in config.items():
        if key not in expconfig:
            expconfig[key] = value
    return expconfig

def build_detectron2_config(experiment_config):
    """
        Build detectron2 config from our config file
    """

    det2_cfgFile = experiment_config['detectorName']
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(det2_cfgFile))
    cfg.DATASETS.TRAIN = ("fiftyone_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(det2_cfgFile)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = experiment_config['batchSize']  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = experiment_config['learningRate']  # pick a good LR
    cfg.SOLVER.MAX_ITER = experiment_config['numSteps']    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES =  experiment_config['num_classes'] # only has one class (Vehicle registration plate). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.SEED = experiment_config['seed'] ## set seed for reproducibility
    cfg.OUTPUT_DIR = experiment_config['outputDir']
    cfg.SOLVER.CHECKPOINT_PERIOD = experiment_config['save_after_steps']
    cfg.MODEL.ROI_HEADS.BBOX_REG_LOSS_TYPE = experiment_config['bbox_loss']
    cfg.INPUT.MIN_SIZE_TRAIN = experiment_config['imageMinMaxSizes'][0]
    cfg.INPUT.MAX_SIZE_TRAIN = experiment_config['imageMinMaxSizes'][1]
    cfg.INPUT.MIN_SIZE_TEST = experiment_config['imageMinMaxSizes'][0]
    cfg.INPUT.MAX_SIZE_TEST = experiment_config['imageMinMaxSizes'][1]
    cfg.INPUT.CROP.ENABLED = experiment_config['cropEnabled']
    cfg.INPUT.CROP.SIZE = experiment_config['cropSize']
    return cfg