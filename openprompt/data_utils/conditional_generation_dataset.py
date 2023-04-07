# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all Conditional Generation tasks.
"""

from openprompt.data_utils.utils import InputExample
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor
from tqdm import tqdm

class WebNLGProcessor(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.json".format(split))
        with open(path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        guid_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            if split.lower() == "train":
                for sent in sents:
                    if sent["comment"] == 'good':
                        full_tgt_lst.append(sent["lex"])
                        full_src_lst.append(temp_triples)
                        full_rela_lst.append(rela_lst)
            else:
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)
                temp = []
                for sent in sents:
                    if sent["comment"] == 'good':
                        temp.append(sent["lex"])
                full_tgt_lst.append("\n".join(temp))

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        if split.lower() == "train":
            for i, (src, tgt) in enumerate(zip(full_src_lst, full_tgt_lst)):
                example = InputExample(guid=str(i), text_a=src, tgt_text=tgt)
                examples.append(example)
        else:
            for i, (src, tgt) in enumerate(zip(full_src_lst, full_tgt_lst)):
                example = InputExample(guid=str(i), text_a=src, tgt_text=tgt)
                examples.append(example)
        return examples


    def get_src_tgt_len_ratio(self,):
        pass


class CSQAProcessor(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python
        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = os.path.join(root_dir, "datasets/Reasoning")
        dataset_name = "csqa"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 9741
        assert len(valid_dataset) == 1221
        assert len(test_dataset) == 1140
        assert train_dataset[0].text_a == "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?"
        assert train_dataset[0].text_b == ""
        assert train_dataset[0].tgt_text == "A"
        assert train_dataset[0].meta == {'choices': [{'label': 'A', 'text': 'ignore'}, {'label': 'B', 'text': 'enforce'}, {'label': 'C', 'text': 'authoritarian'}, {'label': 'D', 'text': 'yell at'}, {'label': 'E', 'text': 'avoid'}], 'choices_text': '(A) ignore\n(B) enforce\n(C) authoritarian\n(D) yell at\n(E) avoid'}

    """

    split2file = {
        "train": "train_rand_split.jsonl",
        "dev": "dev_rand_split.jsonl",
        "test": "test_rand_split_no_answers.jsonl"
    }

    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, self.split2file[split])
        i = 0
        with open(path) as f:
            for line in f.readlines():
                if line.strip():
                    data = json.loads(line)
                    context = data["question"]["stem"]
                    choices = data["question"]["choices"]
                    tgt_text = data.get("answerKey", None)
                    example = InputExample(guid=str(i), text_a=context, tgt_text=tgt_text, meta={"choices": choices})
                    examples.append(example)
                    i += 1
        return examples


    def get_src_tgt_len_ratio(self,):
        pass


class UltraChatProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_path: str) -> List[InputExample]:
        examples = []
        j = 0
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                if line.strip():
                    data = json.loads(line)
                    id_ = data["id"]
                    dialogue = data["data"]
                    tags = [i for _ in range(len(dialogue)//2) for i in ["User", "Assistant"]]
                    for i in range(0, len(dialogue), 2):
                        tgt_text = dialogue[i+1]
                        context = dialogue[:i+1]
                        context = zip(tags[:i+1], context)
                        context = [": ".join(item) for item in context]
                        example = InputExample(guid=str(j), text_a="", tgt_text=tgt_text, meta={"context": context})
                        examples.append(example)
                        j += 1
        return examples


    def get_src_tgt_len_ratio(self,):
        pass

PROCESSORS = {
    "webnlg_2017": WebNLGProcessor,
    "webnlg": WebNLGProcessor,
    "csqa": CSQAProcessor,
    "ultrachat": UltraChatProcessor
    # "e2e": E2eProcessor,
    # "dart" : DartProcessor,
}
