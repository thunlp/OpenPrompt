
import random
import numpy as np
import torch

def set_seed(config):
    """set seed for reproducibility

    Args:
        global config object
    """
    reproduce_config = config.reproduce
    random_seed = reproduce_config.seed
    numpy_seed = reproduce_config.seed
    torch_seed = reproduce_config.seed
    cuda_seed = reproduce_config.seed
    if reproduce_config.random_seed >= 0:
        random_seed = reproduce_config.random_seed
    if reproduce_config.numpy_seed >= 0:
        numpy_seed = reproduce_config.numpy_seed
    if reproduce_config.torch_seed >= 0:
        torch_seed = reproduce_config.torch_seed
    if reproduce_config.cuda_seed >= 0:
        cuda_seed = reproduce_config.cuda_seed

    if random_seed >= 0:
        random.seed(random_seed)
    if numpy_seed >= 0:
        np.random.seed(numpy_seed)
    if torch_seed >= 0:
        torch.manual_seed(torch_seed)
    if config.environment.num_gpus and cuda_seed >= 0:
        torch.cuda.manual_seed_all(cuda_seed)
    



    
    