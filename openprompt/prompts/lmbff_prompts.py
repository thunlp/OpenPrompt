from typing import List, Optional, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
from .manual_template import ManualTemplate
from .manual_verbalizer import ManualVerbalizer
from transformers.data.processors.utils import InputExample
from typing import List, Optional, Dict


class LMBFFTemplate(ManualTemplate):
    """
    This is a special template used only for earch of template in LM-BFF using `T5ForConditionalGeneration`. For example, a template could be ``<text_a> <extra_id_0> <label> <extra_id_1>``, where ``<label>`` is replaced by label_words in verbalizer in `wrap_one_example` method.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        verbalizer (:obj:`ManualVerbalizer`): A verbalizer to provide label_words.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        label_token (:obj:`str`, optional): The special token that needs to be replaced by label_words. Default to ``<label>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 verbalizer: ManualVerbalizer,
                 text: Optional[List[str]] = None,
                 mask_token: str = '<mask>',
                 label_token: str = '<label>',
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.text = text
        self.verbalizer = verbalizer
        self.label_token = label_token
    
    def wrap_one_example(self, 
                         example: InputExample) -> List[Dict]:
        wrapped_example = super().wrap_one_example(example)

        # replace label_token with label words
        if self.verbalizer.label_words and example.label:
            if self.label_token in wrapped_example[0][0]['text']:
                wrapped_example[0][0]['text'][wrapped_example[0][0]['text'].index(self.label_token)] = self.verbalizer.label_words[example.label]

        return wrapped_example
