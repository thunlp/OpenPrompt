import itertools
import warnings
from typing import Union, List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import inspect
from collections import namedtuple
from math import ceil

def signature(f):
    r"""Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.
    
    Args:
        f (:obj:`function`) : the function to get the input arguments.
    
    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s

    """
    sig = inspect.signature(f)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None
    defaults = [
        p.default for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not p.empty
    ] or None
    argspec = namedtuple('Signature', ['args', 'defaults',
                                        'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords) 

def round_list(l: List[float], max_sum:int):
    r"""round a list of float e.g. [0.2,1.5, 4.5]
    to [1,2,4] # ceil and restrict the sum to `max_sum`
    used into balanced truncate.
    """
    s = 0
    for idx, i in enumerate(l):
        i = ceil(i)
        if s <= max_sum:
            s += i
            if s <= max_sum:
                l[idx] = i
            else:
                l[idx] = i - (s - max_sum)
        else:
            l[idx] = int(0)
    assert sum(l) == max_sum


class TokenizerWrapper:
    def __init__(self,
                 max_seq_length: int,
                 tokenizer,
                 truncate_method: Optional[str] = 'tail',
                 create_token_type_ids: Optional[str] = False,
                 **kwargs):
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        if truncate_method=='tail':
            self.truncate_fct = self.truncate_from_tail
        elif truncate_method=='head':
            self.truncate_fct = self.truncate_from_head
        elif truncate_method == 'balanced':
            self.truncate_fct = self.balanced_truncate
        else:
            raise NotImplementedError

        self.create_token_type_ids = create_token_type_ids

        self.template_mask_token = '<mask>'
        self.template_eos_token = '<eos>'
        self.template_bos_token = '<bos>'
        self.template_sep_token = '<sep>'
        self.template_cls_token = '<cls>'
        self.template_pad_token = '<pad>'

        self.mask_token_map = {self.template_mask_token: self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else ''}
        self.eos_token_map = {self.template_eos_token: self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else ''}
        self.bos_token_map = {self.template_bos_token: self.tokenizer.bos_token if hasattr(self.tokenizer, 'bos_token') else ''}
        self.sep_token_map = {self.template_sep_token: self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else ''}
        self.cls_token_map = {self.template_cls_token: self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else ''}
        self.pad_token_map = {self.template_pad_token: self.tokenizer.pad_token if hasattr(self.tokenizer, 'pad_token') else ''}

        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def truncate_rate(self,):
        r"""Using this function, one can easily identify how many sentence has be truncated, thus help the user to choose a better thresthold for chunking.
        """
        if self.total_passed_sentences==0:
            return None
        else:
            return self.num_truncated_sentences/self.total_passed_sentences

    @property
    def special_tokens_maps(self,) -> Dict:
        r"""This need to be specified in specific language model
        """
        if not hasattr(self, "_special_tokens_map"):
            _special_tokens_map = {}
            for attrname in self.__dict__.keys():
                if attrname.endswith('_token_map'):
                    _special_tokens_map.update(getattr(self, attrname))
        return  _special_tokens_map

    def tokenize_with_mask(self,
                            wrapped_example: List[Dict],
                            ):
        raise NotImplementedError

    def tokenize_without_mask(self,
                            wrapped_example: List[Dict],
                            ):
        raise NotImplementedError

    @staticmethod
    def balanced_truncate(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        '''truncate the inputs with balance, number of cut tokens is proportional to the part's length.
        '''
        shortenable_lens = [len(parts) if parts[0]==1 else 0
                                  for parts in input_dict['shortenable_ids']]
        total_shortenable_len = sum(shortenable_lens)
        num_tokens_to_truncate_each_part = [part_len/total_shortenable_len*num_tokens_to_truncate
                                                for part_len in shortenable_lens]
        round_list(num_tokens_to_truncate_each_part, num_tokens_to_truncate)

        truncated_example = defaultdict(list)
        for key in input_dict:
            parts = input_dict[key]
            for num_tokens_to_truncate_part, part in zip(num_tokens_to_truncate_each_part, parts):
                truncated_example[key].append(part[:len(part)-num_tokens_to_truncate_part])
        return truncated_example

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the rear
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts[::-1]):
                if len(part) == 0: # to prevent some part are empty after tokenization
                    continue
                if shortenable_ids[-1-i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[-1-i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def truncate_from_head(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the head
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts):
                if shortenable_ids[i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    @staticmethod
    def padding(input_dict: Dict,
                max_len: int, pad_id_for_inputs: int=0, pad_id_for_others: int=0) -> None:
        for key, value in input_dict.items():
            if (len(input_dict[key]) > max_len):
                raise ValueError(f'''Truncated seq length of '{key}' still greater than max length {max_len}."\
                    "One possible reason is that no enough shortenable parts in template. Try adding {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:
                input_dict[key].extend([pad_id_for_inputs]*(max_len-len(value)))
            else:
                input_dict[key].extend([pad_id_for_others]*(max_len-len(value)))
        return input_dict


    def add_special_tokens(self, encoder_inputs):
            # add special tokens
        for key in encoder_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                                                        encoder_inputs[key])
            else:
                special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key]))
                with_special_tokens = np.array(self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key]))
                if key in ["soft_token_ids"]: # TODO maybe more than this
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens).tolist() # use 0 as special
                else:
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens - special_tokens_mask*100).tolist() # use -100 as special
        return encoder_inputs

    def truncate(self, encoder_inputs):
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences+=1
        if num_tokens_to_truncate>0:
            self.num_truncated_sentences += 1
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                          num_tokens_to_truncate=num_tokens_to_truncate)
        return encoder_inputs