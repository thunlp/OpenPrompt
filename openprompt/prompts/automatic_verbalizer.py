from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.data_utils.data_utils import InputFeatures
from openprompt import Verbalizer
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger



class AutomaticVerbalizer(Verbalizer):
    r""" 
    This implementation is slightly different from the original code in that
    1). we allow re-selecting the verbalizer after a fixed training steps. 
    The original implementation only performs one step selection after getting
    the inital logits on the training data. To adopt their implementation,
    please only do optimize() after the first pass of training data.

    2). We strictly follows the probility calculation in Equation (3) in the
    paper, which take softmax over the logits.

    3). We do not implements the ``combine_patterns'' if-branch. Since it's
    not a pure verbalizer type, and doesn't yield much improvement. However, 
    it can be achieve by using EnsembleTrainer to pass text wrapped by 
    multiple templates together with this verbalizer. 
    
    We use a probs_buffer to store the probability :math:`q_{P,t}(1|\mathbf{x})` that to be used in later verbalizer selection, 
    and a label_buffer to store the label :math:`y` that to be used in later verbalizer selection.

    Args:
        num_candidates (:obj:`int`, optional): the number of candidates for further selection based on Section 4.1
        label_word_num_per_class (:obj:`int`, optional): set to be greater than 1 to support Multi-Verbalizers in Section 4.2
        num_searches (:obj:`int`, optional): Maximnum number of label_words search. After reaching this number, the verbalizer will use the same label_words as the previous iterations. 
        search_id (:obj:`int`, optional): the id of current search, used to determine when to stop label words searching.
        score_fct (:obj:`str`, optional): the scoring function of label words selection. ``llr'' means log likelihood ratio, corresponding to Equation (7); ``ce'' means cross entropy, corresponding to Equation (6). As the paper points out, ``llr'' is significantly better than 'ce', we only keep it to match the original code.
        balance (:obj:`book`, optional): whether to perform normalization of unbalanced training dataset, as Equation (5).
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer = None,
                 num_candidates: Optional[int]= 1000,
                 label_word_num_per_class: Optional[int] = 1,
                 num_searches: Optional[int] = 1,
                 score_fct: Optional[str] = 'llr',
                 balance: Optional[bool] = True,
                 num_classes: Optional[bool] = None,
                 classes: Optional[List[str]] = None,
                 init_using_split: Optional[str] = "train",
                 **kwargs):
        super().__init__(num_classes=num_classes, tokenizer = tokenizer, classes=classes)
        self.num_candidates = num_candidates
        self.label_word_num_per_class = label_word_num_per_class
        self.probs_buffer, self.labels_buffer = None, None
        assert num_searches > 0, "You requires the verbalizer to perform {} searches. Invalid.".format(num_searches)
        self.num_searches = num_searches
        self.search_id = 0
        self.accumulate_step = 0 # currently not used, to support not epoch-level optimize.
        self.accumulate = True # A flag to indicate whether to 
                               # accumulate examples for optimization.
                               # set to False after finish optimization.
        self.score_fct = score_fct
        self.balance = balance
        self.init_using_split = init_using_split

    def register_buffer(self, logits, labels):
        r'''
        
        Args:
            logits (:obj:`torch.Tensor`): 
            labels (:obj:`List`): 
        '''
        
        logits = F.softmax(logits.detach(),dim=-1)
        labels = labels.detach()
        if self.probs_buffer is None :
            self.probs_buffer = logits
            self.labels_buffer = labels
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels])

    def process_logits(self, logits: torch.Tensor, **kwargs):
    
        if self.accumulate: # inherit from nn.Module, only store buffer in training mode.
            self.accumulate_step+=1
            self.register_buffer(logits, kwargs['batch']['label'])

        if hasattr(self, "label_words_ids"): # TODO the content in this "if" is same as super()
            # project
            label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

            # aggreate
            if label_words_logits.dim()>2:
                label_logits = self.aggregate(label_words_logits)
            else:
                label_logits = label_words_logits
            return label_logits

        else:
            return torch.randn((logits.size(0), self.num_classes), requires_grad=True).to(logits.device)
        
    def project(self,
                logits: torch.Tensor,
                **kwargs, # TODO 
                ) -> torch.Tensor:
        r"""When this verbalizer hasn't perform optimize(), it has no 
        ``label_words_ids``, thus will give random predictions, and should
        have no connection to the model to give (miss-leading) grads.
        
        Args:
            logits (:obj:`torch.Tensor`): The original logits over the vocabulary.
            
        Returns:
            :obj:`torch.Tensor`: The projected logits of label words.
        """
        label_words_logits = logits[:, self.label_words_ids]
        return label_words_logits


    def optimize(self):
        pass
        
        
    def optimize_to_initialize(self):
        r"""This is an epoch-level optimize. If used in batch-level like an ordinary
        gradient descend optimizer, the result may not be very satisfying since the accumated
        examples (i.e., the probs_buffer and the labels_buffer) are not enough if the batchsize
        is small.
        """
        if self.search_id < self.num_searches:
            self.label_words_ids = self._find_verbalizer(words_per_label=self.label_word_num_per_class, 
                                                         num_candidates=self.num_candidates,
                                                         score_fct=self.score_fct,
                                                         balance=self.balance)
            self.probs_buffer, self.labels_buffer = None, None
            self.search_id += 1
            if self.search_id == self.num_searches: # finish optimization
                self.accumulate = False
        else:
            logger.info("Verbalizer's max num_searches reached, use the previous label words.")
        self._show_verbalizer()
        
            
    def _show_verbalizer(self):
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in self.label_words_ids]
        logger.info("Verbalizer is {}".format(tokens))


    def _find_verbalizer(self, words_per_label: int = 1, num_candidates: int = 1000, balance: bool = True,
                         score_fct: str = 'llr'):

        # if score_fct == 'random':
        #      return {label: random.sample(self.word2idx.keys(), words_per_label) for label in self.labels}
        logger.info("Finding verbalizer ...")
        probs = self.probs_buffer
        labels = self.labels_buffer
        candidates = self._get_candidates(num_candidates=num_candidates, probs=probs, labels=labels)
        label_words =  self._get_top_words(probs=probs, candidates=candidates, balance=balance, words_per_label=words_per_label,
                                    score_fct=score_fct)
        return label_words

    def _get_candidates(self, 
                        num_candidates: int,
                        probs: torch.Tensor,
                        labels: torch.Tensor,
                        ) -> Dict[str, List[str]]:
        if num_candidates <= 0:
            return [torch.arange(self.vocab_size) for label_id in range(self.num_classes)]

        log_probs = torch.log(probs+1e-15)
        candidate_ids = []
        for label_id in range(self.num_classes):
            label_mask = (labels==label_id).to(torch.float).unsqueeze(-1)
            score = torch.sum(log_probs * label_mask, dim=0)
            candidate_id = torch.argsort(score, descending=True)[:num_candidates]
            candidate_ids.append(candidate_id)
        return candidate_ids
    
    def _get_top_words(self, 
                       probs: torch.Tensor,
                       candidates: List[torch.Tensor],
                       balance: bool = True,
                       words_per_label: int = 10,
                       score_fct: Optional[str] = 'llr'):
        label_words_ids = []
        for label_id in range(self.num_classes):
            label_mask = (self.labels_buffer==label_id).to(torch.float)
            probs_per_label = probs[:, candidates[label_id]]
            if score_fct == 'llr':
                s = self._log_likelihood_ratio(probs_per_label, label_mask, balance)
            elif score_fct == 'ce':
                s = self._cross_entropy(probs_per_label, label_mask, balance)
            else:
                raise ValueError(f"Score function '{score_fct}' not implemented")
            sorted_ids  = torch.argsort(s, descending=True)[:words_per_label]
            selected_ids = candidates[label_id][sorted_ids]
            label_words_ids.append(selected_ids)
        label_words_ids = torch.vstack(label_words_ids)
        return label_words_ids

    def _log_likelihood_ratio(self, probs, label_mask, balance):
        if balance:
            scale_factor =  torch.sum(label_mask) / torch.sum(1 - label_mask) \
                            * (1-label_mask).unsqueeze(-1)
        else:
            scale_factor = (1-label_mask).unsqueeze(-1)
        label_mask = label_mask.unsqueeze(-1)

        pos_score = torch.sum(torch.log(probs+1e-15) * label_mask, dim=0) - torch.sum(torch.log(1 - probs + 1e-15) * label_mask, dim=0)
        neg_score = torch.sum(torch.log(1 - probs +1e-15) * scale_factor, dim=0) - torch.sum(torch.log(probs+1e-15) * scale_factor, dim=0)
        return pos_score + neg_score
    
    def _cross_entropy(self, probs, label_mask, balance):
        if balance:
            scale_factor =  torch.sum(label_mask) / torch.sum(1 - label_mask) \
                            * (1-label_mask).unsqueeze(-1)
        else:
            scale_factor = (1-label_mask).unsqueeze(-1)
        label_mask = label_mask.unsqueeze(-1)

        pos_score = torch.sum(torch.log(probs+1e-15) * label_mask, dim=0)
        neg_score = torch.sum(torch.log(1 - probs +1e-15) * scale_factor, dim=0)
        return pos_score + neg_score
    
    def from_file(self,
                  path: str, 
                  choice: Optional[int] = 0 ):
        raise NotImplementedError("This verbalizer is learned and can't be set from file.")
