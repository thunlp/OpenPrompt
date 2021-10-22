import torch

def model_to_device(model, config):
    r"""
    model: the model to be wrapped
    config: the environment subconfig.
    """
    import os
    if "CUDA_VISIBLE_DEVICES" not in os.environ and config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config.cuda_visible_devices])
    if config.num_gpus>1:
        local_rank_device = "cuda:{}".format(config.local_rank)
        model = model.to(local_rank_device)
        model = torch.nn.parallel.DataParallel(model, output_device=local_rank_device)
    elif config.num_gpus>0:
        model = model.cuda()
    else:
        pass
    return model