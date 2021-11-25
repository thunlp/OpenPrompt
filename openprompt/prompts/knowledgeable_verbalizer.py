import os
from openprompt.prompts.manual_template import ManualTemplate
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.data_utils import InputFeatures
import re
from openprompt.prompts.manual_verbalizer import ManualVerbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger



class KnowledgeableVerbalizer(ManualVerbalizer):
    r"""
    This is the implementation of knowledeagble verbalizer, which uses external knowledge to expand the set of label words.
    This class inherit the ``ManualVerbalizer`` class.
    
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`classes`): The classes (or labels) of the current task.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        max_token_split (:obj:`int`, optional): 
        verbalizer_lr (:obj:`float`, optional): The learning rate of the verbalizer optimization.
        candidate_frac (:obj:`float`, optional): 
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer = None,
                 classes: Sequence[str] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 max_token_split: Optional[int] = -1,
                 verbalizer_lr: Optional[float]=5e-2,
                 candidate_frac: Optional[float]=0.5,
                 **kwargs):
        super().__init__(classes=classes, prefix=prefix, multi_token_handler=multi_token_handler, tokenizer=tokenizer, **kwargs)
        self.max_token_split = max_token_split
        self.verbalizer_lr = verbalizer_lr
        self.candidate_frac = candidate_frac

    def on_label_words_set(self):
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ' '.
        """
        new_label_words = []
        for words in label_words:
            new_label_words.append([prefix + word.lstrip(prefix) for word in words])
        return new_label_words
        
    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more one token. 
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if self.max_token_split>0  and len(ids) > self.max_token_split: 
                    # in knowledgebale verbalizer, the labelwords may be very rare, so we may 
                    # want to remove the label words which are not recogonized by tokenizer.
                    logger.warning("Word {} is split into {} (>{}) tokens: {}. Ignored.".format(word, \
                                    len(ids), self.max_token_split,
                                    self.tokenizer.convert_ids_to_tokens(ids)))
                    continue
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        
        

        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        self.label_words_weights = nn.Parameter(torch.zeros(self.num_classes, max_num_label_words), requires_grad=True)
        logger.info("Num of label words for each label: {}".format(self.label_words_mask.sum(-1).cpu().tolist()))
        self.verbalizer_optimizer = torch.optim.AdamW(self.parameters(), lr=self.verbalizer_lr)


    def register_calibrate_logits(self, logits: torch.Tensor):
        r"""For Knowledgeable Verbalizer, it's nessessory to filter the words with has low prior probability.
        Therefore we re-compute the label words after register calibration logits.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits
        cur_label_words_ids = self.label_words_ids.data.cpu().tolist()
        rm_calibrate_ids = set(torch.argsort(self._calibrate_logits)[:int(self.candidate_frac*logits.shape[-1])].cpu().tolist())

        new_label_words = []
        for i_label, words_ids_per_label in enumerate(cur_label_words_ids):
            new_label_words.append([])
            for j_word, word_ids in enumerate(words_ids_per_label):
                if j_word >= len(self.label_words[i_label]):
                    break
                if len((set(word_ids).difference(set([0]))).intersection(rm_calibrate_ids)) == 0:
                    new_label_words[-1].append(self.label_words[i_label][j_word])
        self.label_words = new_label_words
        self.to(self._calibrate_logits.device)
    

    def project(self,
                 logits: torch.Tensor,
                 **kwargs,
                 ) -> torch.Tensor:
        r"""The return value if the normalized (sum to 1) probs of label words. 
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logots of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.
        
        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words. 
        """
        label_words_weights = F.softmax(self.label_words_weights-10000*(1-self.label_words_mask), dim=-1)
        label_words_logits = (label_words_logits * self.label_words_mask * label_words_weights).sum(-1)
        return label_words_logits

        
    def from_file(self,
                  path: str,
                  separator: Optional[str] = ',', 
                  ):
        r"""Load the predefined  label words from verbalizer file 
        """
        with open(path, 'r') as fin:
            label_words = fin.readlines()
            label_words = [words_per_label.strip().split(separator) for words_per_label in label_words]
        num_classes = len(label_words)
        assert num_classes==self.num_classes, 'number of classes in the verbalizer file\
                                           does not match the predefined num_classes.'
        self.label_words = self.add_prefix(label_words, self.prefix)
        return self
    
    def optimize(self,):
        self.verbalizer_optimizer.step()
        self.verbalizer_optimizer.zero_grad()
