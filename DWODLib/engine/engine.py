## Here we get trainer object

from detectron2.engine import DefaultTrainer, DefaultPredictor

## trainer class
class Trainer(object):
    def __init__(self, cfg, experiment_config):
        self.dt = DefaultTrainer(cfg) ## default trainer object
        
        ## let's freeze what we gotta freeze
        if experiment_config['freeze_backbone']:
            self.freeze_backbone()

        if experiment_config['freeze_RPN']:
            self.freeze_RPN()

    def freeze_backbone(self):
        """
            This will freeze backbone and RPN
        """
        
        for name, param in self.dt.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False

    def freeze_RPN(self):
        """
            This will freeze RPN
        """
        
        for name, param in self.dt.model.named_parameters():
            if 'proposal_generator' in name:
                param.requires_grad = False



class Predictor(object):
    def __init__(self, cfg):
        self.dp = DefaultPredictor(cfg) ## default predictor object

