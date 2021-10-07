
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from openprompt.plms.utils import TokenizerWrapper
from typing import List, Dict
from collections import defaultdict

class MLMTokenizerWrapper(TokenizerWrapper):
    add_input_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    
    @property
    def mask_token(self):
        return self.tokenizer.mask_token
    
    @property
    def mask_token_ids(self):
        return self.tokenizer.mask_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doens't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example
        encoder_inputs = defaultdict(list)
        add_prefix_space = " " # Whether adding a space before the first word.
        for piece in wrapped_example:
            if piece['loss_ids']==1:
                encode_text = [self.mask_token_ids]
            elif 'new_token_ids' in piece and piece['new_token_ids']!=0:
                encode_text = [0] # can be replace by any token, since these token will use their own embeddings
            else:
                encode_text = self.tokenizer.encode(add_prefix_space+piece['text'], add_special_tokens=False)
            add_prefix_space = " "
            encoding_length = len(encode_text)
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key!='text':
                    encoder_inputs[key].append([piece[key]]*encoding_length)
        
        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        # create special input ids
        encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        encoder_inputs['token_type_ids'] = [0] *len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        encoder_inputs = dict(encoder_inputs) # convert defaultdict to dict
        return encoder_inputs
    










