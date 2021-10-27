
import os
from openprompt.utils.logging import logger



from openprompt.data_utils.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class MixedTemplate(ManualTemplate):
    """Mixed of manual token, trainable token and trainable that initialized with given hard token

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        prompt_encoder_type (:obj:`str`): head above the embedding layer of trainable tokens. Can be ``mlp`` or ``None``
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        mixed_token_start (:obj:`str`, optional): The special mark for the start of the trainable token. Default to ``{``
        mixed_token_end (:obj:`str`, optional): The special mark for the end of the trainable token. Default to ``}``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    registered_inputflag_names = ["trainable_token_ids", "loss_ids", "shortenable_ids"]


    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 prompt_encoder_type: Optional[str] = None,
                 mask_token: Optional[str] = '<mask>',
                 trainable_token: Optional[str] = '<train>',
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.trainable_token = self.trainable_token
        self.prompt_encoder_type = prompt_encoder_type
        self.text = text
    
    def get_default_trainable_token_ids(self) -> List[int]:
        r"""get the trainable token indices for the template
        e.g. when self.text is ['<train>1', '<text_a>', '<train>2:is', '<train>1', '<mask>', '.'],
        output is [1, 0, 2, 1, 0, 0]
        """
        idx = []
        num_soft_token = 0
        for token in self.text:
            if token.startswith(self.soft_):
                num_soft_token += 1
                idx.append(num_soft_token)
            else:
                idx.append(0)
        return idx

    def on_text_set(self):
        """
        when template text was set, generate parameters needed for soft-prompt
        """
        self.num_soft_token = sum([token.startswith(self.soft_token) for token in self.text])
        self.generate_parameters()

    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        self.soft_embedding = nn.Embedding(1+self.num_soft_token, self.embedding_size)
        count = 0
        for token in self.text:
            if token.startswith(self.soft_token):
                count += 1
                orig = token.split(self.soft_token)[1]
                if orig == "":
                    # raise ValueError("hard prompt not given")
                    continue
                token_ids = self.tokenizer(" " + orig, add_special_tokens=False)["input_ids"] # TODO no prefix space option
                if len(token_ids) > 1:
                    logger.warning("""soft prompt's hard prompt {} tokenize to more than one tokens: {}
                        By default we use the first token""".format(orig, self.tokenizer.convert_ids_to_tokens(token_ids)))
                self.soft_embedding.weight.data[count, :] = self.raw_embedding.weight.data[token_ids[0], :].clone().detach().requires_grad_(True)# TODO check this

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        raw_embeds = self.raw_embedding(batch['input_ids'])
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds, raw_embeds)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
