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

if __name__ == "__main__":
    test_AgnewsProcessor()