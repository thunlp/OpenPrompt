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

import os
from typing import List
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

# logger = log.get_logger(__name__)

class SSTDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = int(linelist[1])
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)
        return examples

class SNLIDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ['entailment', 'neutral', 'contradiction']
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                guid = "%s-%s" % (split, idx)
                label = linelist[-1]
                text_a = linelist[7]
                text_b = linelist[8]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
                examples.append(example)
        return examples

PROCESSORS = {
    "sst-2": SSTDataProcessor,
    "snli": SNLIDataProcessor
}