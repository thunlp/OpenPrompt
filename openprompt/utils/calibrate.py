
from yacs.config import CfgNode
from openprompt.data_utils import FewShotSampler
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample
from openprompt.pipeline_base import PromptDataLoader, PromptModel, PromptForClassification
from typing import *
import torch
from tqdm import tqdm

# def pmi_calibrate(prompt_model: PromptModel,context_dataloader, max_seq_length: int) -> torch.Tensor:
#     r"""Pmi calibrate. See `Paper <https://arxiv.org/pdf/2104.08315.pdf>`_
    
#     Args:
#         prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
#         max_seq_length: (:obj:`int`): the truncation parameters for dataloader.
#     """
#     prompt = prompt_model.prompt
#     tokenizer = prompt_model.tokenizer
#     virtual_dataset = [InputExample(guid='000', text_a='', text_b='')]
#     context_dataloader = PromptDataLoader(virtual_dataset, prompt, tokenizer, max_seq_length=max_seq_length, batch_size=len(virtual_dataset),device=prompt_model.device)
#     for batch in context_dataloader:
#         logits = prompt_model.forward_without_verbalize(batch)
#         logits = logits[torch.where(batch.loss_ids>0)]
#     return logits.mean(dim=0)

def calibrate(prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
    r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
    
    Args:
        prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
        dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
    
    Return:
        (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
    """
    all_logits = []
    prompt_model.eval()
    for batch in tqdm(dataloader,desc='ContextCali'):
        batch = batch.to(prompt_model.device)
        logits = prompt_model.forward_without_verbalize(batch)
        all_logits.append(logits.detach())
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits.mean(dim=0)


# def calibrate(model: PromptForClassification, calibrate_method: str=None, config: CfgNode =None , train_dataset: Optional[List]=None, valid_dataset: Optional[List]=None):
#     r"""Calibrate the PromptForClassification model. Select and run the calibrate using the global config node.

#     Args:
#         model (:obj:`PromptForClassification`): the PromptForClassification model.
#         config (:obj:`CfgNode`): The global config node.
#         train_dataset: (:obj:`List`): the training dataset, if use the training dataset to do contextualized calibrate.
#         valid_dataset: (:obj:`List`): the valid dataset, if use the valid dataset to do contextualized calibrate.
#     """
#     if config.calibrate == "pmi_calibrate":
#         calibrate_logits = pmi_calibrate(model, max_seq_length=config.dataloader.max_seq_length)
#         model.register_calibrate_logits(calibrate_logits)
#     elif config.calibrate_type == "contextualized_calibrate":
#         if config.contextualized_calibrate.use_split == "train":
#             context_dataset = train_dataset
#         elif config.contextualized_calibrate.use_split == "valid":
#             context_dataset = valid_dataset
#         elif config.contextualized_calibrate.use_split is None and config.contextualized_calibrate.num_example is not None:
#             sampler = FewShotSampler(num_examples_total=config.contextualized_calibrate.num_example,
#                                 also_sample_dev=False)
#             context_dataset = sampler(train_dataset)
#         calibrate_logits = contextualized_calibrate(model, context=context_dataset, max_seq_length=config.dataloader.max_seq_length)
#         model.register_calibrate_logits(calibrate_logits)

