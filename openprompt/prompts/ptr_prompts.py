
import json
from openprompt.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Verbalizer
from openprompt.prompts import One2oneVerbalizer, PtuningTemplate

class PTRTemplate(PtuningTemplate):
    """
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        soft_token (:obj:`str`, optional): The special token for soft token. Default to ``<soft>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text:  Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                ):
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         prompt_encoder_type="mlp",
                         text=text,
                         placeholder_mapping=placeholder_mapping)


class PTRVerbalizer(Verbalizer):
    """
    In `PTR <https://arxiv.org/pdf/2105.11259.pdf>`_, each prompt has more than one ``<mask>`` tokens.
    Different ``<mask>`` tokens have different label words.
    The final label is predicted jointly by these label words using logic rules.

    Args: 
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
        label_words (:obj:`Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]`, optional): The label words that are projected by the labels.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Sequence[str] = None,
                 num_classes: Optional[int] = None,
                 label_words: Optional[Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]] = None,
                ):
        super().__init__(tokenizer = tokenizer, classes = classes, num_classes = num_classes)
        self.label_words = label_words

    def on_label_words_set(self):
        """
        Prepare One2oneVerbalizer for each `<mask>` seperately
        """
        super().on_label_words_set()

        self.num_masks = len(self.label_words[0])
        for words in self.label_words:
            if len(words) != self.num_masks:
                raise ValueError("number of mask tokens for different classes are not consistent")
        self.sub_labels = [
            list(set([words[i] for words in self.label_words]))
            for i in range(self.num_masks)
        ] # [num_masks, label_size of the corresponding mask]

        self.verbalizers = nn.ModuleList([
            One2oneVerbalizer(tokenizer=self.tokenizer, label_words=labels, post_log_softmax = False)
            for labels in self.sub_labels
        ]) # [num_masks]

        self.label_mappings = nn.Parameter(torch.LongTensor([
            [labels.index(words[j]) for words in self.label_words]
            for j, labels in enumerate(self.sub_labels)
        ]), requires_grad=False) # [num_masks, label_size of the whole task]

    def process_logits(self,
                       logits: torch.Tensor, # [batch_size, num_masks, vocab_size]
                       batch: Union[Dict, InputFeatures],
                       **kwargs):
        """
        1) Process vocab logits of each `<mask>` into label logits of each `<mask>`

        2) Combine these logits into a single label logits of the whole task

        Args:
            logits (:obj:`torch.Tensor`): vocab logits of each `<mask>` (shape: `[batch_size, num_masks, vocab_size]`)

        Returns:
            :obj:`torch.Tensor`: logits (label logits of whole task (shape: `[batch_size, label_size of the whole task]`))
        """
        each_logits = [ # logits of each verbalizer
            self.verbalizers[i].process_logits(logits = logits[:, i, :], batch = batch, **kwargs)
            for i in range(self.num_masks)
        ] # num_masks * [batch_size, label_size of the corresponding mask]

        label_logits = [
            logits[:, self.label_mappings[j]]
            for j, logits in enumerate(each_logits)
        ]

        logsoftmax = nn.functional.log_softmax(sum(label_logits), dim=-1)

        if 'label' in batch: # TODO not an elegant solution
            each_logsoftmax = [ # (logits of each label) of each mask
                nn.functional.log_softmax(logits, dim=-1)[:, self.label_mappings[j]]
                for j, logits in enumerate(each_logits)
            ] # num_masks * [batch_size, label_size of the whole task]

            return logsoftmax + sum(each_logsoftmax) / len(each_logits) # [batch_size, label_size of the whole task]

        return logsoftmax
