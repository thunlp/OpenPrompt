
from json import decoder

from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.plms.utils import TokenizerWrapper
from typing import List, Dict, Optional
from collections import defaultdict
from openprompt.utils.logging import logger

class T5TokenizerWrapper(TokenizerWrapper):
    r"""
    Given wrapped example, e.g. A fun movie ! it is <mask> and <mask> .
    The input tokens is:  A fun movie ! it is <extra_id_0> and <extra_id_1> . </s>
    The decoder tokens is: <pad> <extra_id_0> <extra_id_1> </s>
    The expected output is <extra_id_0> good interesting </s>
    """
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 decoder_max_length: Optional[int] = 128,
                 decode_from_pad: Optional[bool] = True,
                 predict_eos_token: Optional[bool] = False,
                 **kwargs):
        super().__init__(max_seq_length=max_seq_length, tokenizer=tokenizer,truncate_method=truncate_method)
        self.decoder_max_length = decoder_max_length
        self.decode_from_pad = decode_from_pad
        self.predict_eos = predict_eos_token
        if self.create_token_type_ids:
            logger.warning("token_type_ids is not valid in T5. will be depreciated.")

    def mask_token(self,i):
        return self.tokenizer.additional_special_tokens[i]
        
        
    def mask_token_ids(self, i ):
        return self.tokenizer.additional_special_tokens_ids[i]

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials


    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doens't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example
        
        if teacher_forcing:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0
        
        decoder_input_ids = []
        loss_ids =[]
        
        for piece_id, piece in enumerate(wrapped_example):
            if piece['text'] == self.template_mask_token:
                if teacher_forcing:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    loss_ids.append(0)
                    encode_text = [self.mask_token_ids(num_mask_token_used)] 
                    tgt_text_ids = self.tokenizer.encode(" " + tgt_text[num_mask_token_used], add_special_tokens=False)
                    decoder_input_ids.extend(tgt_text_ids)
                    loss_ids.extend([1] * len(tgt_text_ids))
                    # decoder_input_ids.append(self.mask_token_ids(num_mask_token_used+1))
                else:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    encode_text = [self.mask_token_ids(num_mask_token_used)] 
                    # decoder_input_ids.append(self.mask_token_ids(num_mask_token_used+1))
                    # loss_ids[-1] = 1 # shift loss_ids
                    loss_ids.append(1)
                num_mask_token_used += 1
            else:
                if piece['text'] in self.special_tokens_maps.keys():
                    to_replace = self.special_tokens_maps[piece['text']]
                    if to_replace is not None:
                        piece['text'] = to_replace
                    else:
                        raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

                if 'soft_token_ids' in piece and piece['soft_token_ids']!=0:
                    encode_text =  [0] # can be replace by any token, since these token will use their own embeddings
                else: 
                    encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)
                
            encoding_length = len(encode_text)
            
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text', 'loss_ids']:
                    encoder_inputs[key].append([piece[key]]*encoding_length)

        # decoder input ids
        decoder_inputs = {'decoder_input_ids': decoder_input_ids, 'loss_ids':loss_ids}
        decoder_inputs = self.truncate_decoder_inputs(decoder_inputs)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)

        # create special input ids
        encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        all_input_ids = {**encoder_inputs, **decoder_inputs}
        return all_input_ids

    def truncate_decoder_inputs(self, inputs):
        if self.decode_from_pad:
            inputs['decoder_input_ids'].insert(0, self.tokenizer.pad_token_id)
            inputs['loss_ids'].insert(0, 0)
        
        for key in inputs:
            inputs[key] = inputs[key][:self.decoder_max_length - 1]
        
        if self.predict_eos:
            inputs['decoder_input_ids'].append(self.tokenizer.eos_token_id)
            inputs['loss_ids'].append(1)
        inputs = self.padding(inputs, max_len = self.decoder_max_length, pad_id_for_inputs=self.tokenizer.pad_token_id)
        return inputs

class T5LMTokenizerWrapper(TokenizerWrapper):
    r"""
    The tokenizerwrapper is for the t5-lm-adapted version proposed by
    `The Power of Scale for Parameter-Efficient Prompt Tuning <https://arxiv.org/abs/2104.08691>`_
    Since this model is a autogressive language model fashion, it only support generation from the 
    and of the text. 

    Given wrapped example, e.g. A fun movie ! it is {"mask"}
    The encoder input is :  A fun movie ! it is </s>  
    (Note that </s> is added in T5 encoder inputs, this will yield better result compared to not using </s>)
    The decoder input is : <pad> <extra_id_0> </s>
    The expected output is : good
    Under teacher forcing mode, the decoder input is   <pad> <extra_id_0>  good </s>, where good and </s> requires loss
    """
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 decoder_max_length: Optional[int] = 128,
                 decode_from_pad: Optional[bool] = True,
                 predict_eos_token: Optional[bool] = False,
                 **kwargs):
        super().__init__(max_seq_length=max_seq_length, tokenizer=tokenizer,truncate_method=truncate_method)
        self.decoder_max_length = decoder_max_length
        self.decode_from_pad = decode_from_pad
        self.predict_eos = predict_eos_token
        if self.create_token_type_ids:
            logger.warning("token_type_ids is not valid in T5. will be depreciated.")

    def mask_token(self,i):
        return self.tokenizer.additional_special_tokens[i]
        
        
    def mask_token_ids(self, i ):
        return self.tokenizer.additional_special_tokens_ids[i]

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials


    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doens't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example
        if teacher_forcing: 
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0
        
        decoder_input_ids = []
        loss_ids =[]
        
        for piece_id, piece in enumerate(wrapped_example):
            if piece['text'] == self.template_mask_token:
                if teacher_forcing:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    loss_ids.append(0)
                    encode_text = [] 
                    tgt_text_ids = self.tokenizer.encode(" " + tgt_text[num_mask_token_used], add_special_tokens=False)
                    decoder_input_ids.extend(tgt_text_ids)
                    loss_ids.extend([1] * len(tgt_text_ids))
                else:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    encode_text = [] # not add extra_id_0 to input_ids
                    loss_ids.append(1)
                break
            else:
                if piece['text'] in self.special_tokens_maps.keys():
                    to_replace = self.special_tokens_maps[piece['text']]
                    if to_replace is not None:
                        piece['text'] = to_replace
                    else:
                        raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

                if 'soft_token_ids' in piece and piece['soft_token_ids']!=0:
                    encode_text =  [0] # can be replace by any token, since these token will use their own embeddings
                else: 
                    encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)
                
            encoding_length = len(encode_text)
            
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text', 'loss_ids']:
                    encoder_inputs[key].append([piece[key]]*encoding_length)

        # decoder input ids
        decoder_inputs = {'decoder_input_ids': decoder_input_ids, 'loss_ids':loss_ids}
        decoder_inputs = self.truncate_decoder_inputs(decoder_inputs)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)

        # create special input ids
        encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        all_input_ids = {**encoder_inputs, **decoder_inputs}
        return all_input_ids

    def truncate_decoder_inputs(self, inputs):
        if self.decode_from_pad:
            inputs['decoder_input_ids'].insert(0, self.tokenizer.pad_token_id)
            inputs['loss_ids'].insert(0, 0)
        
        for key in inputs:
            inputs[key] = inputs[key][:self.decoder_max_length - 1]
        
        inputs['decoder_input_ids'].append(self.tokenizer.eos_token_id)
        if self.predict_eos:
            inputs['loss_ids'].append(1)
        else:
            inputs['loss_ids'].append(0)
        inputs = self.padding(inputs, max_len = self.decoder_max_length)
        return inputs




class CPM2TokenizerWrapper(TokenizerWrapper):
    r"""
    Given wrapped example, e.g. A fun movie ! it is <mask> and <mask> .
    The input tokens is:  A fun movie ! it is <s_0> and <s_1> . <eod>
    The decoder tokens is: <s> <s_0> <s_1> <eod>
    The expected output is: <s_0> good <s_1> interesting </s>
    Another solution is add verbalizer label words into the labels, thus generating
    'interesting' given 'good'. However, it is not implemented in this version.
    """
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 decoder_max_length: Optional[int] = 128,
                 decode_from_start: Optional[bool] = True,
                 predict_eos_token: Optional[bool] = False,
                 **kwargs):
        super().__init__(max_seq_length=max_seq_length, tokenizer=tokenizer,truncate_method=truncate_method)
        self.decoder_max_length = decoder_max_length
        self.decode_from_start = decode_from_start
        self.predict_eos = predict_eos_token
        if self.create_token_type_ids:
            logger.warning("token_type_ids is not valid in T5. will be depreciated.")

    def mask_token(self,i):
        return self.tokenizer.additional_special_tokens[i]
        
        
    def mask_token_ids(self, i ):
        return self.tokenizer.additional_special_tokens_ids[i]

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials


    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doens't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example
        
        if teacher_forcing:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0
        
        decoder_input_ids = []
        loss_ids =[0]
        
        for piece_id, piece in enumerate(wrapped_example):
            if piece['text'] == self.template_mask_token:
                if teacher_forcing:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    encode_text = [self.mask_token_ids(num_mask_token_used)] 
                    tgt_text_ids = self.tokenizer.encode(" " + tgt_text[num_mask_token_used], add_special_tokens=False)
                    decoder_input_ids.extend(tgt_text_ids)
                    loss_ids.extend([1] * len(tgt_text_ids))
                    # decoder_input_ids.append(self.mask_token_ids(num_mask_token_used+1))
                    loss_ids.append(1)
                else:
                    decoder_input_ids.append(self.mask_token_ids(num_mask_token_used))
                    encode_text = [self.mask_token_ids(num_mask_token_used)] 
                    # decoder_input_ids.append(self.mask_token_ids(num_mask_token_used+1))
                    loss_ids[-1] = 1 # shift loss_ids
                    loss_ids.append(0)
                num_mask_token_used += 1
            else:
                if piece['text'] in self.special_tokens_maps.keys():
                    to_replace = self.special_tokens_maps[piece['text']]
                    if to_replace is not None:
                        piece['text'] = to_replace
                    else:
                        raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

                if 'soft_token_ids' in piece and piece['soft_token_ids']!=0:
                    encode_text =  [0] # can be replace by any token, since these token will use their own embeddings
                else: 
                    encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)
                
            encoding_length = len(encode_text)
            
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text', 'loss_ids']:
                    encoder_inputs[key].append([piece[key]]*encoding_length)

        # decoder input ids
        decoder_inputs = {'decoder_input_ids': decoder_input_ids, 'loss_ids':loss_ids}
        decoder_inputs = self.truncate_decoder_inputs(decoder_inputs)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)

        # create special input ids
        encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        all_input_ids = {**encoder_inputs, **decoder_inputs}
        return all_input_ids

    def truncate_decoder_inputs(self, inputs):
        if self.decode_from_start:
            inputs['decoder_input_ids'].insert(0, self.tokenizer.sod_token_id)
            inputs['loss_ids'].insert(0, 0)
        
        for key in inputs:
            inputs[key] = inputs[key][:self.decoder_max_length - 1]
        
        if self.predict_eos:
            inputs['decoder_input_ids'].append(self.tokenizer.eos_token_id)
            inputs['loss_ids'].append(1)
        inputs = self.padding(inputs, max_len = self.decoder_max_length, pad_id_for_inputs=self.tokenizer.pad_token_id)
        return inputs
            


         


    
