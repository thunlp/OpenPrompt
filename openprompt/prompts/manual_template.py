
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template

class ManualTemplate(Template):
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[List[str]] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         placeholder_mapping=placeholder_mapping)
        self.text = text
    
    def on_text_set(self):
        """
        when template text was set
        
        1. parse text
        """

        self.text = self.parse_text(self.text)