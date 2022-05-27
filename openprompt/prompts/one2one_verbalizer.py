import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger



class One2oneVerbalizer(Verbalizer):
    r"""
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.
    This class restrict the use of label words to one words per label. For a verbalzer with less constraints,
    please use Basic ManualVerbalizer.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`classes`): The classes (or labels) of the current task.
        num_classes (:obj:`int`): Optional. The number of classes of the verbalizer. Only one of `classes` and `num_classes` should be used.
        label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer. (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 num_classes: Optional[int] = None,
                 classes: Optional[List] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
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
        if isinstance(label_words[0], list):
            assert max([len(w) for w in label_words]) ==  1, "Providing multiple label words, you should use other verbalizers instead."
            label_words = [w[0] for w in label_words]

        for word in label_words:
            if word.startswith("<!>"):
                new_label_words.append(word.split("<!>")[1])
            else:
                new_label_words.append(prefix + word)

        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        words_ids = []
        for word in self.label_words:
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

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the label words set.
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        return label_words_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs /= norm
        return label_words_probs









