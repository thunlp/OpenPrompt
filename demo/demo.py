import sys
sys.path.append(".")

import tqdm, os
def nop(it, *a, **k): return it
tqdm.tqdm = nop
os.environ.get("TQDM_CONFIG", '')

from transformers import logging
logging.set_verbosity(logging.CRITICAL)

from openprompt.utils.logging import logger

logger.setLevel(logging.CRITICAL)

import torch

def color(text, color="\033[35m"): # or \033[32m
    return color+text+"\033[0m"

def input_selection(what, lis, delimiter="\n"):
    print(f"Select a {color(what)}: ")
    for idx, item in enumerate(lis):
        print(f"    {idx+1}.", item, end=delimiter)
    if delimiter != '\n': print()
    idx = int(input(f"Enter a number between 1 to {len(lis)} :   "))-1
    print()
    return lis[idx]

def input_enter(what):
    res = input(f"Enter the {color(what)}: ")
    print()
    return res

def progress_print(text):
    print(text, "...")

if __name__ == "__main__":
    print()
    progress_print(f"This demo is powered by {color('OpenPrompt')}")
    print()

    from openprompt.data_utils import InputExample
    text = input_enter("text")
    '''
    Albert Einstein was one of the greatest intellects of his time.
    '''
    dataset = [
        InputExample(
            guid = 0,
            text_a = text,
        )
    ]

    from openprompt.plms import get_model_class
    model_class = get_model_class(plm_type = "roberta")
    model_path = "roberta-large"
    bertConfig = model_class.config.from_pretrained(model_path)
    bertTokenizer = model_class.tokenizer.from_pretrained(model_path)
    bertModel = model_class.model.from_pretrained(model_path)

    template = input_enter("Prompt Template")
    '''
        <text_a> It is <mask>
        <text_a> Albert Einstein is a <mask>
        Albert Einstein was born in <mask>
    '''
    from openprompt.prompts import ManualTemplate
    template = ManualTemplate(
        text = template.split(),
        tokenizer = bertTokenizer,
    )

    verbalizer = input_selection("Prompt Verbalizer", [
        'Sentiment Verbalizer',
        'Entity Verbalizer',
        'Knowledge Probing',
    ])
    classes = None
    if verbalizer == 'Knowledge Probing':
        verbalizer = None
        classes = {v:k for k,v in bertTokenizer.get_vocab().items()}
    else:
        label_words = None
        if verbalizer == "Entity Verbalizer":
            label_words = {
                "location-bodiesofwater": ["water"],
                "other-law": ["law"],
                "art-broadcastprogram": ["broadcast", "program"],
                "organization-media/newspaper": ["media", "newspaper"],
                "building-restaurant": ["restaurant"],
                "person-artist/author": ["artist", "author"],
                "art-film": ["film"],
                "other-award": ["award"],
                "location-park": ["park"],
                "event-other": ["event"],
                "organization-government/governmentagency": ["government", "agency"],
                "person-other": ["person"],
                "other-educationaldegree": ["educational", "degree"],
                "organization-education": ["education"],
                "person-director": ["director"],
                "product-game": ["game"],
                "building-sportsfacility": ["sports", "facility"],
                "event-protest": ["protest"],
                "product-car": ["car"],
                "other-language": ["language"],
                "building-airport": ["airport"],
                "organization-other": ["organization"],
                "building-other": ["building"],
                "location-other": ["location"],
                "person-athlete": ["athlete"],
                "organization-showorganization": ["show", "organization"],
                "organization-sportsleague": ["sports", "league"],
                "location-GPE": ["geopolitical"],
                "person-scholar": ["scholar", "scientist"],
                "building-library": ["library"],
                "building-hotel": ["hotel"],
                "location-road/railway/highway/transit": ["road", "railway", "highway", "transit"],
                "art-painting": ["painting"],
                "building-hospital": ["hospital"],
                "event-election": ["election"],
                "art-writtenart": ["written", "art"],
                "organization-religion": ["religion"],
                "organization-company": ["company"],
                "product-train": ["train"],
                "product-ship": ["ship"],
                "event-attack/battle/war/militaryconflict": ["attack", "battle", "war", "military", "conflict"],
                "event-sportsevent": ["sports", "event"],
                "event-disaster": ["disaster"],
                "other-currency": ["currency"],
                "product-weapon": ["weapon"],
                "other-livingthing": ["living"],
                "organization-sportsteam": ["sports", "team"],
                "person-politician": ["politician"],
                "other-god": ["god"],
                "organization-politicalparty": ["political", "party"],
                "art-music": ["music"],
                "art-other": ["art"],
                "person-actor": ["actor"],
                "building-theater": ["theater"],
                "other-biologything": ["biology"],
                "product-software": ["software"],
                "location-island": ["island"],
                "other-medical": ["medical"],
                "other-disease": ["disease"],
                "other-chemicalthing": ["chemical"],
                "product-other": ["product"],
                "product-airplane": ["airplane"],
                "product-food": ["food"],
                "location-mountain": ["mountain"],
                "other-astronomything": ["astronomy"],
                "person-soldier": ["soldier"],
            }
        elif verbalizer == "Sentiment Verbalizer":
            label_words = {
                "positive": ["great", "wonderful", "well", "good", "nice"],
                "negative": ["bad", "terrible", "ugly", "horrible"]
            }
        classes = list(label_words.keys())
        from openprompt.prompts import ManualVerbalizer
        verbalizer = ManualVerbalizer(
            classes = classes,
            label_words = label_words,
            tokenizer = bertTokenizer
        )

    progress_print(f"Incorporating {color('Template')} and {color('Verbalizer')} into a {color('PromptModel')}")
    from openprompt import PromptForClassification
    prompt_model = PromptForClassification(
        template = template,
        model = bertModel,
        verbalizer = verbalizer,
    )

    progress_print("Predicting")
    print()

    from openprompt import PromptDataLoader
    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = bertTokenizer, 
        template = template, 
    )

    prompt_model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to_dict()
            if verbalizer is None:
                logits = prompt_model.forward_without_verbalize(batch)
            else:
                logits = prompt_model(batch)
            pred = torch.argmax(logits, dim=-1)
            pred = pred.tolist()

    if verbalizer == None:
        print(f"{color('Predicition')}: ", bertTokenizer.convert_tokens_to_string(bertTokenizer.convert_ids_to_tokens(pred)[0]))
    else:
        print(f"{color('Predicition')}: ", classes[pred[0]], f"(triggered by label words: {label_words[classes[pred[0]]]})")

    print()
    print()
