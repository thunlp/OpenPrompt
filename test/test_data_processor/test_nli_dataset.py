import os, sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)
from openprompt.data_utils.nli_dataset import PROCESSORS

base_path = os.path.join(root_dir, "datasets")

def test_SNLIProcessor():
    dataset_name = "SNLI"
    dataset_path = os.path.join(base_path, dataset_name)
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

if __name__ == '__main__':
    test_SNLIProcessor()