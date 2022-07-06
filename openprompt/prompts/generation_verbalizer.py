from functools import partial
import json
from openprompt.data_utils.utils import InputExample
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
from openprompt.utils.crossfit_metrics import *
import re


class GenerationVerbalizer(Verbalizer):
    r"""
    This verbalizer is useful when the label prediction is better defined by a piece of input.
    For example, in correference resolution, the tgt_text is a proper noun mentioned in the text.
    There is no fixed mapping between a class label and its label words. This verbalizer
    can be used as verbalizer of ``COPA`` and ``WSC`` datasets in SuperGlue.

    This verbalizer is especially powerful when combined with
    `All NLP Tasks Are Generation Tasks <https://arxiv.org/abs/2103.10360>`_ Paradigm (Also see
    `Crossfit <https://arxiv.org/abs/2104.08835>`_). It can make any piece of text the tgt_text. The tgt_text will then be filled in the `{"mask"}`.

    For example, when label word is ``"good"``, the tgt_text is ``"good"``;

    when label word is ``{"text":"good"}``, the tgt_text is also ``"good"``;

    when label word is ``{"meta":"choice1"}``, the tgt_text is the ``"meta['choice1']"`` field of the ``InputExample``;

    when label word is ``{"meta":"choice1"} {"placeholder", "text_a"} .``, the tgt_text is the ``"meta['choice1']"`` field of the ``InputExample``,
    followed by ``text_a`` field of the ``InputExample``, and then a ``'.'``;

    A use case can be seen in `Tutorial 4.1 <https://github.com/thunlp/OpenPrompt/blob/main/tutorial/4.1_all_tasks_are_generation.py>`_

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        is_rule (:obj:`bool`, optional): When the verbalizer use the rule syntax of MixTemplate.
        label_words (:obj:`dict`, optional): The label words of the generation verbalizer

    Example:
    To use this verbalizer to train the T5 model to predict answer and explanation using two masks.

    When the template (Defined by :obj:`MixedTemplate`) is:
    >>> input_example = InputExample(text_a = "Can fish run?", meta={"answer":"no", "explanation": "The fish have no legs"}, label=0)
    >>> template = "{'placeholder':'text_a'} answer: {'mask'} explanation: {'mask'}"

    The verbalizer can be:
    >>> label_words = {0:["no", "{'meta':'explanation'}"], 1:["yes", "{'meta':'explanation'}"]}
    >>> verbalizer = GenerationVerbalizer(tokenizer, classes=None, is_rule=True, label_words=label_words)




    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List[str]] = None,
                 num_classes: Optional[int] = None,
                 is_rule: Optional[bool] = False,
                 label_words: Optional[dict] = None,
                ):
        if classes is None and label_words is not None:
            classes = list(label_words.keys())
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = ''
        self.is_rule = is_rule
        self.mixed_token_start = "{"
        self.mixed_token_end = "}"

        if label_words is not None: # use label words as an initialization
            self.label_words = label_words

    def wrap_one_example(self, example: InputExample) -> List[Dict]:
        r"""Take an InputExample, and fill the tgt_text with label words
        """
        if not isinstance(self.label_words[example.label], list):
            label_word = [self.label_words[example.label]]
        else:
            label_word = self.label_words[example.label]

        if example.tgt_text is not None:
            logger.warning(f"The example already has tgt_text {example.tgt_text}, and will be filled with new label words, is this intended?")

        if not self.is_rule:
            instance_label_word =  label_word
        else:
            instance_label_word = [i(example) for i in label_word]  #(example)
        if len(instance_label_word) == 1:
            example.tgt_text = instance_label_word[0]
        else:
            example.tgt_text = instance_label_word
        return example


    def on_label_words_set(self):
        r"""
        Process the text into the label words (sometimes a function) according to the syntax of MixedTemplate
        """
        if isinstance(self.label_words[0], list):
            self.label_words = [x[0] for x in self.label_words]

        if self.is_rule:
            for id, label_word in enumerate(self.label_words):
                try:
                    d = self.parse_text(label_word)
                except:
                    raise RuntimeError(f"is_rule={self.is_rule} but label_word: {label_word} can't be converted to object.")
                self.label_words[id] = partial(lambda x, text: self.incorporate_text_example(text, x), text=d)



    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i-1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ''
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        break
                    j = j + 1
                if j == len(text):
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{'+text[i+1:j]+'}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)

        return parsed

    def incorporate_text_example(self,
                                 text,
                                 example: InputExample
                                ):
        text = text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(example.meta[d['meta']])
            elif 'soft' in d:
                raise RuntimeError("soft token not supported in verbalizer") # unused
            elif 'mask' in d:
                raise RuntimeError("mask token not supported in verbalizer")
            elif 'special' in d:
                raise RuntimeError("special token not supported in verbalizer")
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        text = " ".join(text)
        return text








