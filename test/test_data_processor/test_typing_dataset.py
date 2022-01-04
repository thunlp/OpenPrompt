import os, sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)
from openprompt.data_utils.typing_dataset import PROCESSORS

base_path = os.path.join(root_dir, "datasets/Typing")

def test_FewNERDProcessor():
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
