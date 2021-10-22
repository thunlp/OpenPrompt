
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
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[List[str]] = None,
                 mask_token: str = '<mask>',
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.text = text
        

    def wrap_one_example(self, 
                         example: InputExample) -> List[Dict]:
        '''Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value. 
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as whether 
        it's shortenable, whether it's the masked position, whether it's soft token.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.
        
        Args:       example(:obj:`InputExample`):Should have attributes that 
                    are able to be filled in the template.
        Returns:    :obj:`List[Dict]`
        '''
        
        not_empty_keys = example.keys()
        if self.text is None:
            raise ValueError("template text has not been initialized")
        if isinstance(example, InputExample):
            text = self.text.copy()
            for placeholder_token in self.placeholder_mapping:
                for i in range(len(text)):
                    text[i] = text[i].replace(placeholder_token, getattr(example, self.placeholder_mapping[placeholder_token]))
                not_empty_keys.remove(self.placeholder_mapping[placeholder_token]) # this key has been processed, remove
            for key, value in example.meta.items():
                for i in range(len(text)):
                    text[i] = text[i].replace("<meta:"+key+">", value)
            not_empty_keys.remove('meta') # meta has been processed
            # TODO <a!> rstrip punctuation support
            # print(text) # for debug

            keys, values= ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, "get_default_"+inputflag_name):
                    v = getattr(self, "get_default_"+inputflag_name)()
                else:
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialize.
                    Try using template.{} = [...] to initialize
                    or create an method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))
                
                if len(v) != len(text):
                    raise ValueError("Template: len({})={} doesn't match len(text)={}."\
                        .format(inputflag_name, len(v), len(text)))
                values.append(v)
            wrapped_parts_to_tokenize = []
            for piece in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]
        else:
            raise TypeError("InputExample")        


    def process_batch(self, batch: Union[Dict, InputFeatures])->InputFeatures:
        r"""In manual_template, the batch need not to be processed.
        """
        return batch
    
    @property
    def mask_prefix(self):
        if not hasattr(self, '_mask_prefix'):
            mask_ids = [idx for idx, word in enumerate(self.text) if word==self.mask_token]
            self._mask_prefix = ['' if i==0 else ' ' for i in mask_ids]
        return self._mask_prefix
    
    def from_file(self,
                  path: str,
                  choice: int = 0,
                  separator: str = " ",
                  ):
        with open(path, 'r') as fin:
            text = fin.readlines()[choice]
            text = text.strip().split(separator)
        self.text = text
        return self
        