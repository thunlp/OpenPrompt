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
This file contains the logic for loading data for all FewGLUE tasks.
"""


from openprompt.data_utils.utils import InputExample

from openprompt.data_utils.data_processor import DataProcessor
from datasets import load_dataset
from openprompt.utils.logging import logger
from datasets import load_from_disk
import os

HUGGING_FACE_SCRIPTS = 'super_glue' # if you can not connect huggingface in your machine, you can download the scripts manually and change this line.
HUGGING_FACE_SCRIPTS = '../../huggingface_datasets/super_glue'
class SuperglueMultiRCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["No", "Yes"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='multirc', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.multirc")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['paragraph']
        text_b = example['question']
        meta = {"answer": example["answer"]}
        label = int(example['label'])
        guid = "p{}-q{}-a{}".format(example['idx']['paragraph'], example['idx']['question'], example['idx']['answer'])
        return InputExample(guid = guid, text_a = text_a, text_b = text_b, meta = meta, label=label)

class SuperglueBoolQProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["No", "Yes"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='boolq', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.boolq")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['passage']
        text_b = example['question']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a = text_a, text_b = text_b, label=label)

class SuperglueCBProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["Entailment", "Not Entailment", "Neutral"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='cb', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.cb")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a = text_a, text_b = text_b, label=label)


class SuperglueCOPAProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["No", "Yes"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='copa', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.copa")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        choice1 = example['choice1']
        choice2 = example['choice2']
        premise= example['premise']
        question = example['question']

        meta = {}
        meta['choice1'] = choice1
        meta['choice2'] = choice2
        meta['question'] = question
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a = premise, meta=meta, label=label)

class SuperglueRTEProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["Entailment", "Not Entailment"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='rte', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.rte")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a = text_a, text_b = text_b, label=label)

class SuperglueWiCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["Same", "Different"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='wic', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.wic")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {}
        text_a = example["sentence1"]
        text_b = example["sentence2"]
        meta['word'] = example['word']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a=text_a, text_b=text_b, meta=meta, label=label)

class SuperglueWSCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["Different", "Same"]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='wsc', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.wsc")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        modified_text =  example["text"].split()
        indices = sorted([example['span2_index'], example['span1_index']])
        for idx in indices[::-1]:
            modified_text.insert(idx+1, "*")
            modified_text.insert(idx, "*")
        modified_text = " ".join(modified_text)

        meta = {}
        meta['span1_text'] = example['span1_text']
        meta['span2_text'] = example['span2_text']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, text_a=modified_text, meta=meta, label=label)


class SuperglueRecordProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [None]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        try:
            dataset = load_dataset(path=HUGGING_FACE_SCRIPTS, name='record', cache_dir=data_dir, split=split)
        except:
            dataset = load_from_disk(f"{data_dir}/super_glue.record")[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {}
        meta['passage'] = example["passage"].replace("\n", " ")
        meta['query'] = example["query"]
        meta['entities'] = ", ".join(example['entities'])
        if len(example['answers'])>0:
            meta['answers'] = example['answers'][0]
        else:
            meta['answers'] = ''
        guid = "{}".format(example['idx'])
        return InputExample(guid = guid, meta=meta, label=0)


class YahooAnswersTopicsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [
                        "Society & Culture",
                        "Science & Mathematics",
                        "Health",
                        "Education & Reference",
                        "Computers & Internet",
                        "Sports",
                        "Business & Finance",
                        "Entertainment & Music",
                        "Family & Relationships",
                        "Politics & Government",
                    ]

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "train"# "validation"
        try:
            dataset = load_dataset("yahoo_answers_topics",split=split) # If you have network issues, we use the manually downloaded datasets.
        except:
            dataset = load_from_disk(f"{data_dir}")#[split]
            dataset = dataset[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {}
        text_a = example["question_title"]
        text_b = example["question_content"]
        label = int(example['topic'])
        guid = "{}".format(example["id"])
        return InputExample(guid = guid, text_a=text_a, text_b=text_b, label=label)











PROCESSORS = {
    "super_glue.multirc": SuperglueMultiRCProcessor,
    "super_glue.boolq": SuperglueBoolQProcessor,
    "super_glue.cb": SuperglueCBProcessor,
    "super_glue.copa": SuperglueCOPAProcessor,
    "super_glue.rte": SuperglueRTEProcessor,
    "super_glue.wic": SuperglueWiCProcessor,
    "super_glue.wsc": SuperglueWSCProcessor,
    "super_glue.record": SuperglueRecordProcessor,
    "yahoo_answers_topics": YahooAnswersTopicsProcessor
}
