import sys
sys.path.append(".")

import torch

# A simple sentiment analysis example

from openprompt.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
]


from openprompt.plms import get_model_class
model_class = get_model_class(plm_type = "bert")
model_path = "bert-base-cased"
bertConfig = model_class.config.from_pretrained(model_path)
bertTokenizer = model_class.tokenizer.from_pretrained(model_path)
bertModel = model_class.model.from_pretrained(model_path)

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = ["<text_a>", "It", "was", "<mask>"],
    tokenizer = bertTokenizer,
)

from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = bertTokenizer,
)

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = bertModel,
    verbalizer = promptVerbalizer,
)

from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = bertTokenizer, 
    template = promptTemplate, 
)

promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        print(classes[preds])
# preds would be 1, 0 for classes 'positive', 'negative'
