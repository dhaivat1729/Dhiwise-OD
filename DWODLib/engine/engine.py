## Here we get trainer object

from detectron2.engine import DefaultTrainer

## trainer class
class Trainer(object):
    def __init__(self, cfg):
        self.dt = DefaultTrainer(cfg) ## default trainer object

