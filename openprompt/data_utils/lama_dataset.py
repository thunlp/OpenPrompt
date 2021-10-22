# MIT License
# Copyright (c) 2021 THUDM
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This file contains the logic for loading data for LAMA tasks.
"""

import os
import re
import json, csv

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *
import tokenizers
import sys

from transformers.tokenization_utils import PreTrainedTokenizer

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor



class LAMAProcessor(DataProcessor):
    """This dataset is a variant of the original `LAMA <https://github.com/facebookresearch/LAMA>`_ dataset, which adds train and dev split, and was created by `AutoPrompt <https://github.com/ucinlp/autoprompt>`_ .

    The code of this Processor refers to `the data processing phase in P-tuning <https://github.com/THUDM/P-tuning/tree/main/LAMA>`_

    Args:
        model_name (str): PLM model name.
        tokenizer (PreTrainedTokenizer): tokenizer of the corresponding PLM
        vocab_strategy (str): ["original", "share", "lama"]. "original" use the vocab of PLM; "share" use the vocab of LAMA-29k; "lama" use the vocab of LAMA-34k.
        relation_id (str, optional): [description]. Defaults to "P1001".

    Examples: # TODO test needed
    """
    def __init__(self,
                 base_path: str,
                 model_name: str,
                 tokenizer: PreTrainedTokenizer,
                 vocab_strategy: str,
                 relation_id: str = "P1001"
                ):
        super().__init__()
        self.relation_id = relation_id
        self.tokenizer = tokenizer
        path = os.path.join(base_path, "single_relations/{}.jsonl".format(relation_id))
        with open(path, encoding='utf8') as f:
            template = json.loads(f.readline())["template"]
            if 'gpt' in model_name or 'megatron' in model_name: # TODO generalize to all LM kind model
                self.manual_template = re.sub(r'\[Y\].*', '', template.replace('[X]', "<text_a>"))
            else: # TODO generalize to all MLM kind model
                self.manual_template = template.replace("[X]", "<text_a>").replace("[Y]", "<mask>") # dataset defined
            # TODO Seq2Seq support?
        self.label_mapping = tokenizer.get_vocab()
        self.allowed_vocab_ids = [self.label_mapping[vocab] for vocab in self._get_allowed_vocab(model_name, vocab_strategy, base_path)]

    def _get_allowed_vocab(self, model_name, strategy, base_path):
        if strategy == "original":
            return self.labels
        elif strategy == "share":
            with open(os.path.join(base_path, '29k-vocab.json')) as f:
                shared_vocab = json.load(f)
                if 'gpt' in model_name:
                    return shared_vocab['gpt2-xl']
                elif 'roberta' in model_name or 'megatron' in model_name:
                    return shared_vocab['roberta-large']
                else:
                    assert model_name in shared_vocab
                    return shared_vocab[model_name]
        elif strategy == "lama":
            with open(os.path.join(base_path, '34k-vocab.json')) as f:
                lama_vocab = json.load(f)
                if 'gpt' in model_name:
                    return lama_vocab['gpt2-xl']
                elif 'roberta' in model_name or 'megatron' in model_name:
                    return lama_vocab['roberta-large']
                else:
                    assert model_name in lama_vocab
                    return lama_vocab[model_name]
        else:
            raise ValueError('vocab_strategy must be "original", "share" or "lama"')

    def get_manual_template(self):
        return self.manual_template

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "fact-retrieval/original/{}/{}.jsonl".format(self.relation_id, split)) # TODO oprinal_rob or trex option
        examples = []
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                token_ids = self.tokenizer(" "+example_json["obj_label"], add_special_tokens=False)["input_ids"]
                if len(token_ids) != 1 or token_ids[0] not in self.allowed_vocab_ids:
                    continue
                example = InputExample(guid=str(choicex), text_a=example_json["sub_label"], label=token_ids[0])
                examples.append(example)
        return examples

PROCESSORS = {
    "LAMA": LAMAProcessor, # TODO RENAME this
}
