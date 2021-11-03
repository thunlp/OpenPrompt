from inspect import Parameter
import json
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput

from transformers.models.t5 import  T5ForConditionalGeneration

class SoftVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `WARP <https://aclanthology.org/2021.acl-long.381/>`_

    Args:   
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self, 
                 tokenizer: Optional[PreTrainedTokenizer],
                 plm: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler

        head_name = [n for n,c in plm.named_children()][-1]
        logger.info(f"The LM head named {head_name} was retrieved.")
        self.head = copy.deepcopy(getattr(plm, head_name))
        if isinstance(self.head, torch.nn.Linear):
            self.hidden_dims = self.head.weight.shape[-1]
            self.original_head_last_layer = getattr(plm, head_name)
            self.head = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)
        else: # The head contains multiple layer
            self.head_last_layer_name = [n for n,c in self.head.named_children()][-1]
            head_last_layer = getattr(self.head, self.head_last_layer_name)
            assert isinstance(head_last_layer, torch.nn.Linear), "The retrieved last layer is not a linear layer."

            self.hidden_dims = head_last_layer.weight.shape[-1]
            setattr(self.head, self.head_last_layer_name, torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)) # automatically requires_grad
            self.original_head_last_layer = head_last_layer

        if label_words is not None: # use label words as an initialization
            self.label_words = label_words

        # seperate the last layer with the previous layer.

        
    @property
    def group_parameters_1(self,):
        r"""Include the parameters of head's layer but not the last layer
        In soft verbalizer, note that some heads may contain modules 
        other than the final projection layer. The parameters of these part should be
        optimized (or freezed) together with the plm.
        """
        if isinstance(self.head, torch.nn.Linear):
            return []
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_name not in n]
 
    @property
    def group_parameters_2(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()]
        else:
            return [p for n, p in getattr(self.head, self.head_last_layer_name).named_parameters()]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()
        
    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words
        
    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token. 
        """
        words_ids = []
        for word in self.label_words:
            if isinstance(word, list):
                logger.warning("Label word for a class is a list, only use the first word.")
            word = word[0]
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning("Word {} is split into multiple tokens: {}. \
                    If this is not what you expect, try using another word for this verbalizer" \
                    .format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)

        max_len  = max([len(ids) for ids in words_ids])
        words_ids_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in words_ids]
        words_ids = [ids+[0]*(max_len-len(ids)) for ids in words_ids]
        
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        
        init_data = self.original_head_last_layer.weight[self.label_words_ids,:]*self.label_words_mask.to(self.original_head_last_layer.weight.data.dtype).unsqueeze(-1)
        init_data = init_data.sum(dim=1)/self.label_words_mask.sum(dim=-1,keepdim=True)

        if isinstance(self.head, torch.nn.Linear):
            self.head.weight.data = init_data
            self.head.weight.data.requires_grad=True
        else:
            getattr(self.head, self.head_last_layer_name).weight.data = init_data
            getattr(self.head, self.head_last_layer_name).weight.data.requires_grad=True # To be sure

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 
        """
        label_logits = self.head(hiddens)
        return label_logits
    
    def process_outputs(self, outputs: torch.Tensor, batch: Union[Dict, InputFeatures], **kwargs):
        return self.process_hiddens(outputs)

    def gather_outputs(self, outputs: ModelOutput):
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret


    

        


    
        
