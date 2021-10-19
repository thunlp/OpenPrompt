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
from transformers.data.processors.utils import DataProcessor, InputExample
from openprompt.data_utils import InputExample

# logger = log.get_logger(__name__)

class SSTDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [1, 0]

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "test", train=False)
    
    def _create_examples(self, path: str, set_type: str, train=True) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            if train:
                lines = lines[1:]
            for idx, line in enumerate(lines):
                if train:
                    linelist = line.strip().split('\t')
                    text_a = linelist[0]
                    label = int(linelist[1])
                    guid = "%s-%s" % (set_type, idx)
                    example = InputExample(guid=guid, text_a=text_a, label=label)
                else:
                    linelist = line.strip().split(' ')
                    guid = "%s-%s" % (set_type, idx)
                    label = int(linelist[0])
                    text_a = ' '.join(linelist[1:])
                    example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples

PROCESSORS = {
    "sst-2": SSTDataProcessor,
}