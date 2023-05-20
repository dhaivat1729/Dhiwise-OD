"""
    Override defaultCfg with arguments passed through command line.
"""

import argparse
from ast import literal_eval

def build_dict_from_args(args):
    """
        Build dictionary from arguments passed through command line.
    """
    assert len(args) % 2 == 0, "Arguments must be in key value pairs!"
    args_dict = {}
    for key, value in zip(args[::2], args[1::2]):
        try:
            args_dict[key[2:]] = literal_eval(value)
        except:
            args_dict[key[2:]] = value
        
    return args_dict

def get_args():
    """ Parse arguments from command line"""

    parser = argparse.ArgumentParser(description='Get arguments to overwrite config file')
    """
        Consume all arguments through command line. To modify any key in config file,
        we need to pass the key and value as argument. For example, if we want to modify
        the value of key 'batch_size' in config file, we need to pass the following
        --batch_size "32". 
    """
    
    parser.add_argument('--options', help='Modify config options using the command-line KEY VALUE pair!', default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args().options
    
    if args is not None:
        return build_dict_from_args(args)
    else:
        return {}