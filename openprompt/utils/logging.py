# -*- coding: utf-8 -*-
import logging
import os
import datetime

logger = logging.getLogger()

def config_logger(config):
    r""" Automatic generate log directory for experiments.
    First generate the unique_string of one experiment, if the user
    didn't specify one, according
    to the user-defined keys logging.unique_string_keys.
    Then create the directory.
    """
    if not os.path.exists(config.logging.path_base):
        raise NotADirectoryError("No logging base directory")
    
    # generate unique string
    temp_strs = []
    if config.logging.unique_string is None:
        for item in config.logging.unique_string_keys:
            if item == "datetime":
                continue
            item = item.split(".") # split into sub items.
            subconfig = config
            for key in item:
                try:
                    subconfig = subconfig[key]
                except:
                    raise ValueError("The unique string_key is not a config option ")
            if not isinstance(subconfig, str):
                try:
                    subconfig = str(subconfig)
                except:
                    print("The value of subconfig key {} can't be converted to a string".format(".".join(item)))
                    continue
            
            subconfig = subconfig.split("/")[-1]
            temp_strs.append(subconfig)
        
        if 'datetime' in config.logging.unique_string_keys:
            if config.logging.datetime_format is None:
                config.logging.datetime_format = '%y%m%d%H%M%S'
            time_str = datetime.datetime.now().strftime(config.logging.datetime_format)
            temp_strs.append(time_str)
        config.logging.unique_string = "_".join(temp_strs)
    config.logging.path = os.path.join(config.logging.path_base, config.logging.unique_string)
    
    # create the log directory
    if os.path.exists(config.logging.path):
        if config.logging.overwrite:
            import shutil
            shutil.rmtree(config.logging.path)
            os.mkdir(config.logging.path)
        else:
            raise FileExistsError("Log dir {} exists and can't overwrite!")
    else:
        os.mkdir(config.logging.path)

    kwargs = {}
    kwargs['log_file'] = os.path.join(config.logging.path, "log.txt")
    kwargs['log_file_level'] = getattr(logging, config.logging.file_level)
    kwargs['log_level'] = getattr(logging, config.logging.console_level)
    return kwargs

 
def init_logger(
    config=None,
    log_file=None,
    log_file_level=logging.NOTSET,
    log_level=logging.INFO,
):
    if config is not None:
        kwargs = config_logger(config)
        logger = init_logger(**kwargs)
        return logger
        
    log_format = logging.Formatter("[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s.%(funcName)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger