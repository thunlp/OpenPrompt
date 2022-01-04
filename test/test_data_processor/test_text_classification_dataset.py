import os, sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)
from openprompt.data_utils.text_classification_dataset import PROCESSORS

base_path = os.path.join(root_dir, "datasets/TextClassification")

def test_AgnewsProcessor():
    dataset_name = "agnews"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    trainvalid_dataset = processor.get_train_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert processor.get_num_labels() == 4
    assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
    assert len(trainvalid_dataset) == 120000
    assert len(test_dataset) == 7600
    assert test_dataset[0].text_a == "Fears for T N pension after talks"
    assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
    assert test_dataset[0].label == 2

def test_DBpediaProcessor():
    dataset_name = "dbpedia"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    trainvalid_dataset = processor.get_train_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert processor.get_num_labels() == 14
    assert len(trainvalid_dataset) == 560000
    assert len(test_dataset) == 70000

def test_ImdbProcessor():
    dataset_name = "imdb"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    trainvalid_dataset = processor.get_train_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert processor.get_num_labels() == 2
    assert len(trainvalid_dataset) == 25000
    assert len(test_dataset) == 25000

# def test_AmazonProcessor():
#     dataset_name = "amazon"
#     dataset_path = os.path.join(base_path, dataset_name)
#     processor = PROCESSORS[dataset_name.lower()](dataset_path)
#     trainvalid_dataset = processor.get_train_examples(dataset_path)
#     test_dataset = processor.get_test_examples(dataset_path)

#     assert processor.get_num_labels() == 2
#     assert len(trainvalid_dataset) == 3600000
#     assert len(test_dataset) == 400000

def test_SST2Processor():
    dataset_name = "SST-2"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    train_dataset = processor.get_train_examples(dataset_path)
    dev_dataset = processor.get_dev_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert processor.get_num_labels() == 2
    assert processor.get_labels() == ['0','1']
    assert len(train_dataset) == 6920
    assert len(dev_dataset) == 872
    assert len(test_dataset) == 1821
    assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
    assert train_dataset[0].label == 1
