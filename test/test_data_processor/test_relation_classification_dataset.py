import os, sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)
from openprompt.data_utils.relation_classification_dataset import PROCESSORS

base_path = os.path.join(root_dir, "datasets/RelationClassification")

def test_TACREDProcessor():
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

def test_TACREVProcessor():
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

def test_ReTACREDProcessor():
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

def test_SemEvalProcessor():
    dataset_name = "SemEval"
    dataset_path = os.path.join(base_path, dataset_name)
    processor = PROCESSORS[dataset_name.lower()]()
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