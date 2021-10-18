from typing import OrderedDict
from yacs.config import CfgNode, _merge_a_into_b
import os
from collections import defaultdict 
from .default_config import get_default_config

def get_config_from_file(path):
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(path)
    return cfg

def get_yaml_config(usr_config_path, default_config_path = "config_default.yaml"):
    # get default config
    # cwd = os.path.dirname(__file__)
    # default_config_path = os.path.join(cwd, default_config_path)
    # config = get_config_from_file(default_config_path)
    config = get_default_config()

    # get user config
    usr_config = get_config_from_file(usr_config_path)
    config.merge_from_other_cfg(usr_config)

    config = get_conditional_config(config)
    return config
    

def get_conditional_config(config):
    r"""Extract the config entries that do not have ``parent_config`` key.
    """
    deeper_config = CfgNode(new_allowed=True) # parent key to child node
    configkeys = list(config.keys())
    for key in configkeys:
        if config[key] is not None and 'parent_config' in config[key]:
            deeper_config[key] = config[key]
            config.pop(key)
    
    # breadth first search over all config nodes 
    queue = [config]
    
    while len(queue) > 0:
        v = queue.pop(0)
        ordv = OrderedDict(v.copy())
        while(len(ordv)>0):
            leaf = ordv.popitem()
            if isinstance(leaf[1], str) and \
                leaf[1] in deeper_config.keys():
                retrieved = deeper_config[leaf[1]]
                setattr(config, leaf[1], retrieved)
                if isinstance(retrieved, CfgNode): 
                    # also BFS the newly added CfgNode.
                    queue.append(retrieved)
            elif isinstance(leaf[1], CfgNode):
                queue.append(leaf[1])
    return config


_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_cfg_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
