"""
This file contains the logic for loading data for all RelationClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *

from openprompt.utils.logging import logger

from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor




class TACREDProcessor(DataProcessor):
    """
    `TAC Relation Extraction Dataset (TACRED) <https://nlp.stanford.edu/projects/tacred/>`_ is one of the largest and most widely used datasets for relation classification.

    It was released together with the paper `Position-aware Attention and Supervised Data Improve Slot Filling (Zhang et al. 2017) <https://nlp.stanford.edu/pubs/zhang2017tacred.pdf>`_
    
    This processor is also inherited by :py:class:`TACREVProcessor` and :py:class:`ReTACREDProcessor`. 
    
    Examples:

    ..  code-block:: python 

        from openprompt.data_utils.relation_classification_dataset import PROCESSORS

        base_path = "datasets/RelationClassification"

        dataset_name = "TACRED"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 42
        assert processor.get_labels() == ["no_relation", "org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"]
        assert len(train_dataset) == 68124
        assert len(dev_dataset) == 22631
        assert len(test_dataset) == 15509
        assert train_dataset[0].text_a == 'Tom Thabane resigned in October last year to form the All Basotho Convention -LRB- ABC -RRB- , crossing the floor with 17 members of parliament , causing constitutional monarch King Letsie III to dissolve parliament and call the snap election .'
        assert train_dataset[0].meta["head"] == "All Basotho Convention"
        assert train_dataset[0].meta["tail"] == "Tom Thabane"
        assert train_dataset[0].label == 41
    """

    def __init__(self):
        super().__init__(labels = [
            "no_relation", "org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"
        ])

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.json".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            example_jsons = json.load(f)
            for example_json in example_jsons:
                guid = example_json["id"]
                label = self.get_label_id(example_json["relation"])
                tokens = example_json["token"]
                text_a = " ".join(tokens)
                meta = {
                    "head": " ".join(tokens[example_json["subj_start"]: example_json["subj_end"]+1]),
                    "tail": " ".join(tokens[example_json["obj_start"]: example_json["obj_end"]+1]),
                }

                example = InputExample(guid=guid, text_a=text_a, meta=meta, label=label)
                examples.append(example)
        return examples

class TACREVProcessor(TACREDProcessor):
    """
    `TACRED Revisted (TACREV) <https://github.com/DFKI-NLP/tacrev>`_ is a variant of the TACRED dataset

    It was proposed by the paper `TACRED Revisited: A Thorough Evaluation of the TACRED Relation Extraction Task (Alt et al. 2020) <https://aclanthology.org/2020.acl-main.142.pdf>`_
    
    This processor inherit :py:class:`TACREDProcessor` and can be used similarly
    
    Examples:

    ..  code-block:: python

        from openprompt.data_utils.relation_classification_dataset import PROCESSORS

        base_path = "datasets/RelationClassification"
    
        dataset_name = "TACREV"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)
        assert processor.get_num_labels() == 42
        assert processor.get_labels() == ["no_relation", "org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"]
        assert len(train_dataset) == 68124
        assert len(dev_dataset) == 22631
        assert len(test_dataset) == 15509
    """
    def __init__(self):
        super().__init__()
        self.labels = [
            "no_relation", "org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"
        ]

class ReTACREDProcessor(TACREDProcessor):
    """
    `Re-TACRED <https://github.com/gstoica27/Re-TACRED>`_ is a variant of the TACRED dataset

    It was proposed by the paper `Re-TACRED: Addressing Shortcomings of the TACRED Dataset (Stoica et al. 2021) <https://arxiv.org/pdf/2104.08398.pdf>`_
    
    This processor inherit :py:class:`TACREDProcessor` and can be used similarly
    
    Examples:

    ..  code-block:: python

        from openprompt.data_utils.relation_classification_dataset import PROCESSORS

        base_path = "datasets/RelationClassification"

        dataset_name = "ReTACRED"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)
        assert processor.get_num_labels() == 40
        assert processor.get_labels() == ["no_relation", "org:members", "per:siblings", "per:spouse", "org:country_of_branch", "per:country_of_death", "per:parents", "per:stateorprovinces_of_residence", "org:top_members/employees", "org:dissolved", "org:number_of_employees/members", "per:stateorprovince_of_death", "per:origin", "per:children", "org:political/religious_affiliation", "per:city_of_birth", "per:title", "org:shareholders", "per:employee_of", "org:member_of", "org:founded_by", "per:countries_of_residence", "per:other_family", "per:religion", "per:identity", "per:date_of_birth", "org:city_of_branch", "org:alternate_names", "org:website", "per:cause_of_death", "org:stateorprovince_of_branch", "per:schools_attended", "per:country_of_birth", "per:date_of_death", "per:city_of_death", "org:founded", "per:cities_of_residence", "per:age", "per:charges", "per:stateorprovince_of_birth"]
        assert len(train_dataset) == 58465
        assert len(dev_dataset) == 19584
        assert len(test_dataset) == 13418
    """
    def __init__(self):
        super().__init__()
        self.labels = [
            "no_relation", "org:members", "per:siblings", "per:spouse", "org:country_of_branch", "per:country_of_death", "per:parents", "per:stateorprovinces_of_residence", "org:top_members/employees", "org:dissolved", "org:number_of_employees/members", "per:stateorprovince_of_death", "per:origin", "per:children", "org:political/religious_affiliation", "per:city_of_birth", "per:title", "org:shareholders", "per:employee_of", "org:member_of", "org:founded_by", "per:countries_of_residence", "per:other_family", "per:religion", "per:identity", "per:date_of_birth", "org:city_of_branch", "org:alternate_names", "org:website", "per:cause_of_death", "org:stateorprovince_of_branch", "per:schools_attended", "per:country_of_birth", "per:date_of_death", "per:city_of_death", "org:founded", "per:cities_of_residence", "per:age", "per:charges", "per:stateorprovince_of_birth"
        ]

class SemEvalProcessor(DataProcessor):
    """
    `SemEval-2010 Task 8 <https://aclanthology.org/S10-1006.pdf>`_ is a  a traditional dataset in relation classification.

    It was released together with the paper `SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals (Hendrickx et al. 2010) <https://aclanthology.org/S10-1006.pdf>`_
    
    Examples:

    ..  code-block:: python

        from openprompt.data_utils.relation_classification_dataset import PROCESSORS

        base_path = "datasets/RelationClassification"

        dataset_name = "SemEval"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()](dataset_path)
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)
        assert processor.get_num_labels() == 19
        assert processor.get_labels() == ["Other", "Member-Collection(e1,e2)", "Entity-Destination(e1,e2)", "Content-Container(e1,e2)", "Message-Topic(e1,e2)", "Entity-Origin(e1,e2)", "Cause-Effect(e1,e2)", "Product-Producer(e1,e2)", "Instrument-Agency(e1,e2)", "Component-Whole(e1,e2)", "Member-Collection(e2,e1)", "Entity-Destination(e2,e1)", "Content-Container(e2,e1)", "Message-Topic(e2,e1)", "Entity-Origin(e2,e1)", "Cause-Effect(e2,e1)", "Product-Producer(e2,e1)", "Instrument-Agency(e2,e1)", "Component-Whole(e2,e1)"]
        assert len(train_dataset) == 6507
        assert len(dev_dataset) == 1493
        assert len(test_dataset) == 2717
        assert dev_dataset[0].text_a == 'the system as described above has its greatest application in an arrayed configuration of antenna elements .'
        assert dev_dataset[0].meta["head"] == "configuration"
        assert dev_dataset[0].meta["tail"] == "elements"
        assert dev_dataset[0].label == 18
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Other", "Member-Collection(e1,e2)", "Entity-Destination(e1,e2)", "Content-Container(e1,e2)", "Message-Topic(e1,e2)", "Entity-Origin(e1,e2)", "Cause-Effect(e1,e2)", "Product-Producer(e1,e2)", "Instrument-Agency(e1,e2)", "Component-Whole(e1,e2)", "Member-Collection(e2,e1)", "Entity-Destination(e2,e1)", "Content-Container(e2,e1)", "Message-Topic(e2,e1)", "Entity-Origin(e2,e1)", "Cause-Effect(e2,e1)", "Product-Producer(e2,e1)", "Instrument-Agency(e2,e1)", "Component-Whole(e2,e1)"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                label = self.get_label_id(example_json["relation"])
                text_a = " ".join(example_json["token"])
                meta = {
                    "head": example_json["h"]["name"],
                    "tail": example_json["t"]["name"],
                }

                example = InputExample(guid=str(choicex), text_a=text_a, meta=meta, label=label)
                examples.append(example)
        return examples

PROCESSORS = {
    "tacred": TACREDProcessor,
    "tacrev": TACREVProcessor,
    "retacred": ReTACREDProcessor,
    "semeval": SemEvalProcessor,
}
