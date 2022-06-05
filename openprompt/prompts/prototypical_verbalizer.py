from inspect import Parameter
import json
from os import stat
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

class ProtoVerbalizer(Verbalizer):
    r"""
    The implementation of the prototypical verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ This class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        multi_verb (:obj:`str`, optional): `multi` to ensemble with manual verbalizers, `proto` to use only ProtoVerb.
    """
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 lr: Optional[float] = 1e-3,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 multi_verb: Optional[str] = "multi",
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.multi_verb = multi_verb
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.trained = False

        self.hidden_dims = model.config.hidden_size

        self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)

        if label_words is not None: # use label words as an initialization
            self.label_words = label_words
        w = torch.empty((self.num_classes, self.mid_dim))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.group_parameters_proto, lr=self.lr)

    @property
    def group_parameters_proto(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()] + [self.proto]
        else:
            return [p for n, p in self.head.named_parameters()] + [self.proto]

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
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
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

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

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
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            # label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            # label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

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


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        return label_words_logits

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
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def ensemble_logits(self, manual_logits, proto_logits):

        logits = torch.stack([manual_logits, proto_logits])
        logits = logits.permute(1,0,2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s

    def process_outputs(self, outputs: Union[torch.Tensor, torch.Tensor], batch: Union[Dict, InputFeatures], **kwargs):
        manual_logits = self.process_logits(outputs[1])
        if self.trained is False:
            return manual_logits

        proto_logits = self.process_hiddens(outputs[0])
        if self.trained and self.multi_verb == "proto":
            return proto_logits

        return self.ensemble_logits(manual_logits, proto_logits)

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))

    def pcl_loss(self, v_ins):
        # instance-prototype loss

        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        num = sim_mat.shape[1]
        loss = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:,i,:])
            neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
            loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
        loss = loss / (num * self.num_classes * self.num_classes)

        # instance-instance loss

        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        loss = loss + loss_ins

        return loss


    def train_proto(self, model, dataloader, device):
        model.eval()
        embeds = [[] for _ in range(self.num_classes)]
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to("cuda:{}".format(device)).to_dict()
                outputs = model.prompt_model(batch)
                hidden, _ = self.gather_outputs(outputs)
                outputs_at_mask = model.extract_at_mask(hidden, batch)
                for j in range(len(outputs_at_mask)):
                    label = batch['label'][j]
                    embeds[label].append(outputs_at_mask[j])
        embeds = [torch.stack(e) for e in embeds]
        embeds = torch.stack(embeds)

        instance_mean = embeds.mean(1)
        loss = 0.
        for epoch in range(self.epochs):
            x = self.head(embeds)
            self.optimizer.zero_grad()
            loss = self.pcl_loss(x)
            loss.backward()
            self.optimizer.step()
        logger.info("Total epoch: {}. ProtoVerb loss: {}".format(self.epochs, loss))
        self.trained = True








