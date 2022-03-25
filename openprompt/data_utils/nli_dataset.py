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



class SNLIProcessor(DataProcessor):
    """
    `The Stanford Natural Language Inference (SNLI) corpus <https://nlp.stanford.edu/projects/snli/>`_ is a dataset for natural language inference. It is first released in `A large annotated corpus for learning natural language inference (Bowman et al. 2015) <https://nlp.stanford.edu/pubs/snli_paper.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets"

        dataset_name = "SNLI"
        dataset_path = os.path.join(base_path, dataset_name, '16-13')
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 3
        assert processor.get_labels() == ['entailment', 'neutral', 'contradiction']
        assert len(train_dataset) == 549367
        assert len(dev_dataset) == 9842
        assert len(test_dataset) == 9824
        assert train_dataset[0].text_a == 'A person on a horse jumps over a broken down airplane.'
        assert train_dataset[0].text_b == 'A person is training his horse for a competition.'
        assert train_dataset[0].label == 1
    """
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
    "snli": SNLIProcessor
}