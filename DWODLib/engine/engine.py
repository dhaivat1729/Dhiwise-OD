## Here we get trainer object

from detectron2.engine import DefaultTrainer

## trainer class
class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainer = DefaultTrainer(cfg)

