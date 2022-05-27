try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
import argparse
from yacs.config import CfgNode
import sys
from openprompt.utils.utils import check_config_conflicts
from .default_config import get_default_config
from openprompt.utils.logging import logger
import os


def get_config_from_file(path):
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(path)
    return cfg

def get_user_config(usr_config_path, default_config=None):
    if default_config is None:
        config = get_default_config()
    else:
        config = default_config
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

def add_cfg_to_argparser(cfg, parser, prefix=None):
    r"""To support argument parser style in addition to yaml style
    """
    for key in cfg:
        value = cfg[key]
        full_key_name = prefix+"."+key if prefix is not None else key
        if isinstance(value, CfgNode):
            add_cfg_to_argparser(value, parser=parser, prefix=full_key_name)
        else:
            if type(value) in [str, int, float]:
                parser.add_argument("--"+full_key_name, type=type(value), default=value)
            elif type(value) in [tuple, list]:
                parser.add_argument("--"+full_key_name, type=type(value), default=value, nargs="+")
            elif type(value) == bool:
                parser.add_argument("--"+full_key_name, action='store_{}'.format(not value).lower())
            elif type(value) == type(None):
                parser.add_argument("--"+full_key_name, default=None)
            else:
                raise NotImplementedError("The type of config value is not supported")


def update_cfg_with_argparser(cfg, args, prefix=None):
    r"""To support update cfg with command line
    """
    for key in cfg:
        value = cfg[key]
        full_key_name = prefix+"."+key if prefix is not None else key
        if isinstance(value, CfgNode):
            update_cfg_with_argparser(value, args, prefix=full_key_name)
        else:
            v = getattr(args, full_key_name)
            if type(v) != type(value):
                raise TypeError
            if v != value:
                cfg[key] = v
                print("Update key {}, value {} -> {}".format(full_key_name, value, v))


def save_config_to_yaml(config):
    from contextlib import redirect_stdout
    saved_yaml_path = os.path.join(config.logging.path, "config.yaml")
    with open(saved_yaml_path, 'w') as f:
        with redirect_stdout(f): print(config.dump())
    logger.info("Config saved as {}".format(saved_yaml_path))

def get_config():
    parser = argparse.ArgumentParser("Global Config Argument Parser", allow_abbrev=False)
    parser.add_argument("--config_yaml", required=True, type=str, help='the configuration file for this experiment.')
    parser.add_argument("--resume", type=str, help='a specified logging path to resume training.\
           It will fall back to run from initialization if no latest checkpoint are found.')
    parser.add_argument("--test", type=str, help='a specified logging path to test')
    args, _ = parser.parse_known_args()
    config = get_user_config(args.config_yaml)

    add_cfg_to_argparser(config, parser)
    args = parser.parse_args()

    update_cfg_with_argparser(config, args)
    check_config_conflicts(config)
    # print(config)
    return config, args





