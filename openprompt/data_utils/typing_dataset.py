"""
This file contains the logic for loading data for all typing tasks.
# TODO license
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *

from transformers.tokenization_utils import SPECIAL_TOKENS_MAP_FILE

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor



class FewNERDProcessor(DataProcessor):
    """
    `Few-NERD <https://ningding97.github.io/fewnerd/>`_ a large-scale, fine-grained manually annotated named entity recognition dataset

    It was released together with `Few-NERD: Not Only a Few-shot NER Dataset (Ning Ding et al. 2021) <https://arxiv.org/pdf/2105.07464.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.typing_dataset import PROCESSORS

        base_path = "datasets/Typing"

        dataset_name = "FewNERD"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 66
        assert processor.get_labels() == [
            "person-actor", "person-director", "person-artist/author", "person-athlete", "person-politician", "person-scholar", "person-soldier", "person-other",
            "organization-showorganization", "organization-religion", "organization-company", "organization-sportsteam", "organization-education", "organization-government/governmentagency", "organization-media/newspaper", "organization-politicalparty", "organization-sportsleague", "organization-other",
            "location-GPE", "location-road/railway/highway/transit", "location-bodiesofwater", "location-park", "location-mountain", "location-island", "location-other",
            "product-software", "product-food", "product-game", "product-ship", "product-train", "product-airplane", "product-car", "product-weapon", "product-other",
            "building-theater", "building-sportsfacility", "building-airport", "building-hospital", "building-library", "building-hotel", "building-restaurant", "building-other",
            "event-sportsevent", "event-attack/battle/war/militaryconflict", "event-disaster", "event-election", "event-protest", "event-other",
            "art-music", "art-writtenart", "art-film", "art-painting", "art-broadcastprogram", "art-other",
            "other-biologything", "other-chemicalthing", "other-livingthing", "other-astronomything", "other-god", "other-law", "other-award", "other-disease", "other-medical", "other-language", "other-currency", "other-educationaldegree",
        ]
        assert dev_dataset[0].text_a == "The final stage in the development of the Skyfox was the production of a model with tricycle landing gear to better cater for the pilot training market ."
        assert dev_dataset[0].meta["entity"] == "Skyfox"
        assert dev_dataset[0].label == 30
    """
    def __init__(self):
        super().__init__()
        self.labels = [
            "person-actor", "person-director", "person-artist/author", "person-athlete", "person-politician", "person-scholar", "person-soldier", "person-other",
            "organization-showorganization", "organization-religion", "organization-company", "organization-sportsteam", "organization-education", "organization-government/governmentagency", "organization-media/newspaper", "organization-politicalparty", "organization-sportsleague", "organization-other",
            "location-GPE", "location-road/railway/highway/transit", "location-bodiesofwater", "location-park", "location-mountain", "location-island", "location-other",
            "product-software", "product-food", "product-game", "product-ship", "product-train", "product-airplane", "product-car", "product-weapon", "product-other",
            "building-theater", "building-sportsfacility", "building-airport", "building-hospital", "building-library", "building-hotel", "building-restaurant", "building-other",
            "event-sportsevent", "event-attack/battle/war/militaryconflict", "event-disaster", "event-election", "event-protest", "event-other",
            "art-music", "art-writtenart", "art-film", "art-painting", "art-broadcastprogram", "art-other",
            "other-biologything", "other-chemicalthing", "other-livingthing", "other-astronomything", "other-god", "other-law", "other-award", "other-disease", "other-medical", "other-language", "other-currency", "other-educationaldegree",
        ]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "supervised/{}.txt".format(split))
        with open(path, encoding='utf8') as f:
            data = FewNERDProcessor.load_data(f)

            examples = []

            for idx, (xs, ys, spans) in enumerate(data):
                for span in spans:
                    text_a = " ".join(xs)
                    meta = {
                        "entity": " ".join(xs[span[0]: span[1]+1])
                    }
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=self.get_label_id(ys[span[0]][2:]))
                    examples.append(example)

            return examples

    @staticmethod
    def load_data(file):
        data = []
        xs = []
        ys = []
        spans = []

        for line in file.readlines():
            pair = line.split()
            if pair == []:
                if xs != []:
                    data.append((xs, ys, spans))
                xs = []
                ys = []
                spans = []
            else:
                xs.append(pair[0])

                tag = pair[-1]
                if tag != 'O':
                    if len(ys) == 0 or tag != ys[-1][2:]:
                        tag = 'B-' + tag
                        spans.append([len(ys), len(ys)])
                    else:
                        tag = 'I-' + tag
                        spans[-1][-1] = len(ys)
                ys.append(tag)
        return data

PROCESSORS = {
    "fewnerd": FewNERDProcessor,
    # "conll2003": Conll2003Processor,
    # "ontonotes5_0": OntoNotes5_0Processor,
    # "bbn": BBNProcessor,
}
