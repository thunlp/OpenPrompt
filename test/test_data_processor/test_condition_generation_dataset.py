import os, sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)
from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

base_path = os.path.join(root_dir, "datasets/CondGen")

def test_WebNLGProcessor():
    dataset_name = "webnlg_2017"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    train_dataset = processor.get_train_examples(dataset_path)
    valid_dataset = processor.get_train_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert len(train_dataset) == 18025
    assert len(valid_dataset) == 18025
    assert len(test_dataset) == 4928
    assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
    assert test_dataset[0].text_b == ""
    assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."

def test_CSQAProcessor():
    base_path = os.path.join(root_dir, "datasets/Reasoning")
    dataset_name = "csqa"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
    train_dataset = processor.get_train_examples(dataset_path)
    valid_dataset = processor.get_dev_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    assert len(train_dataset) == 9741
    assert len(valid_dataset) == 1221
    assert len(test_dataset) == 1140
    assert train_dataset[0].text_a == "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?"
    assert train_dataset[0].text_b == ""
    assert train_dataset[0].tgt_text == "A"
    assert train_dataset[0].meta == {'choices': [{'label': 'A', 'text': 'ignore'}, {'label': 'B', 'text': 'enforce'}, {'label': 'C', 'text': 'authoritarian'}, {'label': 'D', 'text': 'yell at'}, {'label': 'E', 'text': 'avoid'}], 'choices_text': '(A) ignore\n(B) enforce\n(C) authoritarian\n(D) yell at\n(E) avoid'}

if __name__ == "__main__":
    test_CSQAProcessor()