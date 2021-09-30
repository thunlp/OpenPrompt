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

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Sequence

from torch.utils.data import dataset

from openprompt.utils.logging import logger

from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor





class FewGLUEDataProcessor(DataProcessor):
    """Processor for FewGLUE
    """

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, "dev32")

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, "val")


class RteProcessor(FewGLUEDataProcessor):
    """Processor for the RTE data set."""

    def __init__(self):
        super().__init__()
        self.labels = ["entailment", "not_entailment"]

    def get_examples(self, data_dir: str, split: str, hypothesis_name: str = "hypothesis",
                         premise_name: str = "premise") -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = choicex
                label = self.get_label_id(example_json["label"])
                guid = "%s-%s" % (split, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples


class CbProcessor(RteProcessor):
    """Processor for the CB data set."""

    def __init__(self):
        super().__init__()
        self.labels = ["entailment", "contradiction", "neutral"]

class WicProcessor(FewGLUEDataProcessor):
    """Processor for the WiC data set."""

    def __init__(self):
        super().__init__()
        self.labels = [True, False]

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = self.get_label_id(example_json["label"])
                guid = "%s-%s" % (split, idx)
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                meta = {'word': example_json['word']}
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx, meta=meta)
                examples.append(example)
        return examples


class WscProcessor(FewGLUEDataProcessor):
    """Processor for the WSC data set."""

    def __init__(self):
        super().__init__()
        self.labels = [True, False]

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = self.get_label_id(example_json["label"])
                guid = "%s-%s" % (split, idx)
                text_a = example_json['text']
                meta = {
                    'span1_text': example_json['target']['span1_text'],
                    'span2_text': example_json['target']['span2_text'],
                    'span1_index': example_json['target']['span1_index'],
                    'span2_index': example_json['target']['span2_index']
                }

                # the indices in the dataset are wrong for some examples, so we manually fix them
                span1_index, span1_text = meta['span1_index'], meta['span1_text']
                span2_index, span2_text = meta['span2_index'], meta['span2_text']
                words_a = text_a.split()
                words_a_lower = text_a.lower().split()
                words_span1_text = span1_text.lower().split()
                span1_len = len(words_span1_text)

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    for offset in [-1, +1]:
                        if words_a_lower[span1_index + offset:span1_index + span1_len + offset] == words_span1_text:
                            span1_index += offset

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    logger.warning(f"Got '{words_a_lower[span1_index:span1_index + span1_len]}' but expected "
                                   f"'{words_span1_text}' at index {span1_index} for '{words_a}'")

                if words_a[span2_index] != span2_text:
                    for offset in [-1, +1]:
                        if words_a[span2_index + offset] == span2_text:
                            span2_index += offset

                    if words_a[span2_index] != span2_text and words_a[span2_index].startswith(span2_text):
                        words_a = words_a[:span2_index] \
                                  + [words_a[span2_index][:len(span2_text)], words_a[span2_index][len(span2_text):]] \
                                  + words_a[span2_index + 1:]

                assert words_a[span2_index] == span2_text, \
                    f"Got '{words_a[span2_index]}' but expected '{span2_text}' at index {span2_index} for '{words_a}'"

                text_a = ' '.join(words_a)
                meta['span1_index'], meta['span2_index'] = span1_index, span2_index

                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                if split == 'train' and label != 'True':
                    continue
                examples.append(example)

        return examples


class BoolQProcessor(FewGLUEDataProcessor):
    """Processor for the BoolQ data set."""

    def __init__(self):
        super().__init__()
        self.labels = [True, False]

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = self.get_label_id(example_json["label"])
                guid = "%s-%s" % (split, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class CopaProcessor(FewGLUEDataProcessor):
    """Processor for the COPA data set."""

    def __init__(self):
        super().__init__()
        self.labels = [0, 1]

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = self.get_label_id(example_json["label"])
                idx = example_json['idx']
                guid = "%s-%s" % (split, idx)
                text_a = example_json['premise']
                meta = {
                    'choice1': example_json['choice1'],
                    'choice2': example_json['choice2'],
                    'question': example_json['question']
                }
                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                examples.append(example)

        if split == 'train' or split == 'unlabeled':
            mirror_examples = []
            for ex in examples:
                label = "1" if ex.label == "0" else "0"
                meta = {
                    'choice1': ex.meta['choice2'],
                    'choice2': ex.meta['choice1'],
                    'question': ex.meta['question']
                }
                mirror_example = InputExample(guid=ex.guid + 'm', text_a=ex.text_a, label=label, meta=meta)
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            logger.info(f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}...")
        return examples


class MultiRcProcessor(FewGLUEDataProcessor):
    """Processor for the MultiRC data set."""

    def __init__(self):
        super().__init__()
        self.labels = [0, 1]

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = self.get_label_id(answer_json["label"])
                        answer_idx = answer_json["idx"]
                        guid = f'{split}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples


class RecordProcessor(FewGLUEDataProcessor):
    """Processor for the ReCoRD data set."""

    def __init__(self):
        super().__init__()
        self.labels = ["0", "1"]

    @staticmethod
    def get_examples(path, split, seed=42, max_train_candidates_per_question: int = 10) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))

        entity_shuffler = random.Random(seed)

        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json['idx']
                text = example_json['passage']['text']
                entities = set()

                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = text[start:end + 1]
                    entities.add(entity)

                entities = list(entities)

                text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
                questions = example_json['qas']

                for question_json in questions:
                    question = question_json['query']
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = answer_json['text']
                        answers.add(answer)

                    answers = list(answers)

                    if split == 'train':
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{split}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta,
                                                   idx=ex_idx)
                            examples.append(example)

                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{split}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples



PROCESSORS = {
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]

