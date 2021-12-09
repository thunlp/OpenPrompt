from abc import abstractmethod
import json

from transformers.file_utils import ModelOutput
from openprompt.config import convert_cfg_to_dict

from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.utils.utils import signature

from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures, InputExample
import torch
import torch.nn as nn
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer

from openprompt.utils.logging import logger
import numpy as np
import torch.nn.functional as F


class Template(nn.Module):
    r'''
    Base class for all the templates. 
    Most of methods are abstract, with some expections to hold the common methods for all template, such as ``loss_ids``, ``save``, ``load``.

    Args: 
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. 
    '''

    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.placeholder_mapping = placeholder_mapping
        self._in_on_text_set = False

        self.mixed_token_start = "{"
        self.mixed_token_end = "}"


    def get_default_loss_ids(self) -> List[int]:
        '''Get the loss indices for the template using mask.
        e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask"}.'``,
        output is ``[0, 0, 0, 0, 1, 0]``.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]:
            
            - 1 for a masked tokens.
            - 0 for a sequence tokens.
        '''
        return [1 if 'mask' in d else 0 for d in self.text]

    def get_default_shortenable_ids(self) -> List[int]:
        """Every template needs shortenable_ids, denoting which part of the template can be trucate to fit
        the language model's ``max_seq_length``. Default: the input text is shortenable, while the template text and other
        special tokens are not shortenable. 

        e.g. when self.text is ``'{"placeholder": "text_a"} {"placeholder": "text_b", "shortenable": False} {"meta": "word"} is {"mask"}.'``,
        output is ``[1, 0, 0, 0, 0, 0, 0]``.
        
        Returns:
            :obj:`List[int]`: A list of integers in the range ``[0, 1]``:

            - 1 for the input tokens.
            - 0 for the template sequence tokens.
        """
        idx = []
        for d in self.text:
            if 'shortenable' in d:
                idx.append(1 if d['shortenable'] else 0)
            else:
                idx.append(1 if 'placeholder' in d else 0)
        return idx

    def get_default_soft_token_ids(self) -> List[int]:
        r'''
        This function identifies which tokens are soft tokens.

        Sometimes tokens in the template are not from the vocabulary, 
        but a sequence of soft tokens.
        In this case, you need to implement this function

        Raises:
            NotImplementedError: if needed, add ``soft_token_ids`` into ``registered_inputflag_names`` attribute of Template class and implement this method.
        '''
        raise NotImplementedError
    
    def incorporate_text_example(self,
                                 example: InputExample
                                ):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(example.meta[d['meta']])
            elif 'soft' in d:
                text[i] = ''; # unused
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text
    
    def _check_template_format(self, ):
        r"""check whether the template format is correct.
        TODO: add more
        """
        mask_num = 0
        for i, d in enumerate(self.text):
            if 'mask' in d:
                mask_num += 1
        
        if mask_num==0:
            raise RuntimeError(f"'mask' position not found in the template: {self.text}. Please Check!")



    
    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i-1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                mixed_token_cnt = 1 # { {} {} } nested support
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0: break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{'+text[i+1:j]+'}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)

        return parsed

    # @abstractmethod
    def wrap_one_example(self, 
                         example: InputExample) -> List[Dict]:
        r'''Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value. 
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as
        whether it's shortenable, whether it's the masked position, whether it's soft token, etc.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.
        
        Args:
            example (:obj:`InputExample`): An :py:class:`~openprompt.data_utils.data_utils.InputExample` object, which should have attributes that are able to be filled in the template.
       
        Returns:
            :obj:`List[Dict]`: A list of dict of the same length as self.text. e.g. ``[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]``
        '''
        
        if self.text is None:
            raise ValueError("template text has not been initialized")
        if isinstance(example, InputExample):
            text = self.incorporate_text_example(example)

            not_empty_keys = example.keys()
            for placeholder_token in self.placeholder_mapping:
                not_empty_keys.remove(self.placeholder_mapping[placeholder_token]) # placeholder has been processed, remove
            not_empty_keys.remove('meta') # meta has been processed

            keys, values= ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, "get_default_"+inputflag_name):
                    v = getattr(self, "get_default_"+inputflag_name)()
                    setattr(self, inputflag_name, v) # cache 
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
    
    @abstractmethod
    def process_batch(self, batch):
        r"""Template should rewrite this method if you need to process the batch input such as substituting embeddings.
        """
        return batch # not being processed

    def post_processing_outputs(self, outputs):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        return outputs
        
    def save(self,
             path: str,
             **kwargs) -> None:
        r'''
        A save method API.
        
        Args:
            path (str): A path to save your template.
        '''
        raise NotImplementedError

    @property
    def text(self):
        return self._text

    @text.setter 
    def text(self, text):
        self._text = text
        if text is None:
            return
        if not self._in_on_text_set:
            self.safe_on_text_set()
        self._check_template_format()
        # else:
        #     logger.warning("Reset text in on_text_set function. Is this intended?")

    def safe_on_text_set(self) -> None:
        r"""With this wrapper function, setting text inside ``on_text_set()``
            will not trigger ``on_text_set()`` again to prevent endless recursion.
        """
        self._in_on_text_set = True
        self.on_text_set()
        self._in_on_text_set = False
   
    @abstractmethod
    def on_text_set(self):
        r"""
        A hook to do something when template text was set.
        The designer of the template should explictly know what should be down when the template text is set.
        """
        raise NotImplementedError
    
    def from_file(self,
                  path: str,
                  choice: int = 0,
                 ):
        r'''
        Read the template from a local file.

        Args: 
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The id-th line of the file.
        '''
        with open(path, 'r') as fin:
            text = fin.readlines()[choice].rstrip()
            logger.info(f"using template: {text}")
        self.text = text
        return self

    @classmethod
    def from_config(cls,
                    config: CfgNode,
                    **kwargs):
        r"""load a template from template's configuration node. 

        Args:
            config (:obj:`CfgNode`): the sub-configuration of template, i.e. config[config.template]
                        if config is a global config node. 
            kwargs: Other kwargs that might be used in initialize the verbalizer. 
                    The actual value should match the arguments of __init__ functions.
        """

        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        template = cls(**init_dict)
        if hasattr(template, "from_file"):
            if not hasattr(config, "file_path"):
                pass
            else:
                if (not hasattr(config, "text") or config.text is None) and config.file_path is not None:
                    if config.choice is None:
                        config.choice = 0
                    template.from_file(config.file_path, config.choice)
                elif (hasattr(config, "text") and config.text is not None) and config.file_path is not None:
                    raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return template
    




class Verbalizer(nn.Module):
    r'''
    Base class for all the verbalizers. 

    Args: 
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
    '''
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 classes: Optional[Sequence[str]] = None,
                 num_classes: Optional[int] = None,
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None
            # raise AttributeError("No able to configure num_classes")
        self._in_on_label_words_set = False

    @property
    def label_words(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels. 
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        if not hasattr(self, "_label_words"):
            raise RuntimeError("label words haven't been set.")
        return self._label_words
    
    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:
            return
        self._label_words = self._match_label_words_to_label_ids(label_words)
        if not self._in_on_label_words_set:
            self.safe_on_label_words_set()
        # else:
        #     logger.warning("Reset label words in on_label_words_set function. Is this intended?")

    def _match_label_words_to_label_ids(self, label_words): # TODO newly add function after docs written # TODO rename this function
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError("name of classes in verbalizer are differnt from those of dataset")
            label_words = [ # sort the dict to match dataset
                label_words[c]
                for c in self.classes
            ] # length: label_size of the whole task
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
            # logger.info("""
            # Your given label words is a list, by default, the ith label word in the list will match class i of the dataset.
            # Please make sure that they have the same order.
            # Or you can pass label words as a dict, mapping from class names to label words.
            # """)
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return label_words

    def safe_on_label_words_set(self,):
        self._in_on_label_words_set = True
        self.on_label_words_set()
        self._in_on_label_words_set = False

    def on_label_words_set(self,):
        r"""A hook to do something when textual label words were set.
        """
        pass

    @property
    def vocab(self,) -> Dict:
        if not hasattr(self, '_vocab'):
            self._vocab = self.tokenizer.convert_ids_to_tokens(np.arange(self.vocab_size).tolist())
        return self._vocab

    @property
    def vocab_size(self,) -> int:
        return self.tokenizer.vocab_size
    
    @abstractmethod
    def generate_parameters(self, **kwargs) -> List:
        r"""
        The verbalizer can be seen as an extra layer on top of the originial
        pre-trained models. In manual verbalizer, it is a fixed one-hot vector of dimension
        ``vocab_size``, with the position of the label word being 1 and 0 everywhere else. 
        In other situation, the parameters may be a continuous vector over the 
        vocab, with each dimension representing a weight of that token.
        Moreover, the parameters may be set to trainable to allow label words selection.
        
        Therefore, this function serves as an abstract methods for generating the parameters
        of the verbalizer, and must be instantiated in any derived class.

        Note that the parameters need to be registered as a part of pytorch's module to 
        It can be acheived by wrapping a tensor using ``nn.Parameter()``.
        """
        raise NotImplementedError

    def register_calibrate_logits(self, logits: torch.Tensor):
        r"""
        This function aims to register logits that need to be calibrated, and detach the orginal logits from the current graph.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits
        
    def process_outputs(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures],
                       **kwargs):
        r"""By default, the verbalizer will process the logits of the PLM's 
        output. 

        Args:
            logits (:obj:`torch.Tensor`): The current logits generated by pre-trained language models.
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of the data.
        """

        return self.process_logits(outputs, batch=batch, **kwargs)

    def gather_outputs(self, outputs: ModelOutput):
        r""" retrieve useful output for the verbalizer from the whole model ouput
        By default, it will only retrieve the logits

        Args:
            outputs (:obj:`ModelOutput`) The output from the pretrained language model.

        Return:
            :obj:`torch.Tensor` The gathered output, should be of shape (``batch_size``, 
            ``seq_len``, ``any``)
        """
        return outputs.logits
    
    @staticmethod
    def aggregate(label_words_logits: torch.Tensor) -> torch.Tensor:
        r""" To aggregate logits on multiple label words into the label's logits
        Basic aggregator: mean of each label words' logits to a label's logits
        Can be re-implemented in advanced verbaliezer.

        Args:
            label_words_logits (:obj:`torch.Tensor`): The logits of the label words only.

        Return:
            :obj:`torch.Tensor`: The final logits calculated by the label words.
        """
        if label_words_logits.dim()>2:
            return label_words_logits.mean(dim=-1)
        else:
            return label_words_logits


    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        r"""
        Given logits regarding the entire vocab, calculate the probs over the label words set by softmax.
       
        Args:
            logits(:obj:`Tensor`): The logits of the entire vocab.

        Returns:
            :obj:`Tensor`: The probability distribution over the label words set.
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    @abstractmethod
    def project(self,
                logits: torch.Tensor,
                **kwargs) -> torch.Tensor:
        r"""This method receives input logits of shape ``[batch_size, vocab_size]``, and use the 
        parameters of this verbalizer to project the logits over entire vocab into the
        logits of labels words. 

        Args: 
            logits (:obj:`Tensor`): The logits over entire vocab generated by the pre-trained lanuage model with shape [``batch_size``, ``max_seq_length``, ``vocab_size``] 
        
        Returns:
            :obj:`Tensor`: The normalized probs (sum to 1) of each label .
        """
        raise NotImplementedError
    
    def handle_multi_token(self, label_words_logits, mask):
        r"""
        Support multiple methods to handle the multi tokens produced by the tokenizer.
        We suggest using 'first' or 'max' if the some parts of the tokenization is not meaningful.
        Can broadcast to 3-d tensor.
    
        Args:
            label_words_logits (:obj:`torch.Tensor`):
        
        Returns:
            :obj:`torch.Tensor`
        """
        if self.multi_token_handler == "first":
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            label_words_logits = label_words_logits - 1000*(1-mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == "mean":
            label_words_logits = (label_words_logits*mask.unsqueeze(0)).sum(dim=-1)/(mask.unsqueeze(0).sum(dim=-1)+1e-15)
        else:
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits
    
    @classmethod
    def from_config(cls, 
                    config: CfgNode, 
                    **kwargs):
        r"""load a verbalizer from verbalizer's configuration node. 

        Args:
            config (:obj:`CfgNode`): the sub-configuration of verbalizer, i.e. ``config[config.verbalizer]``
                        if config is a global config node. 
            kwargs: Other kwargs that might be used in initialize the verbalizer. 
                    The actual value should match the arguments of ``__init__`` functions.
        """

        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs} if config is not None else kwargs
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer = cls(**init_dict)
        if hasattr(verbalizer, "from_file"):
            if not hasattr(config, "file_path"):
                pass
            else:
                if (not hasattr(config, "label_words") or config.label_words is None) and config.file_path is not None:
                    if config.choice is None:
                        config.choice = 0
                    verbalizer.from_file(config.file_path, config.choice)
                elif (hasattr(config, "label_words") and config.label_words is not None) and config.file_path is not None:
                    raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return verbalizer
    
    def from_file(self,
                  path: str, 
                  choice: Optional[int] = 0 ):
        r"""Load the predefined label words from verbalizer file.
        Currently support three types of file format:
        1. a .jsonl or .json file, in which is a single verbalizer 
        in dict format.
        2. a .jsonal or .json file, in which is a list of verbalizers in dict format
        3.  a .txt or a .csv file, in which is the label words of a class are listed in line, 
        seperated by commas. Begin a new verbalizer by an empty line.
        This format is recommended when you don't know the name of each class.

        The details of verbalizer format can be seen in :ref:`How_to_write_a_verbalizer`. 

        Args: 
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The choice of verbalizer in a file containing
                             multiple verbalizers.
        
        Returns:
            Template : `self` object
        """
        if path.endswith(".txt") or path.endswith(".csv"):
            with open(path, 'r') as f:
                lines = f.readlines()
                label_words_all = []
                label_words_single_group = []
                for line in lines:
                    line = line.strip().strip(" ")
                    if line == "":
                        if len(label_words_single_group)>0:
                            label_words_all.append(label_words_single_group)
                        label_words_single_group = []
                    else:
                        label_words_single_group.append(line)
                if len(label_words_single_group) > 0: # if no empty line in the last
                    label_words_all.append(label_words_single_group)
                if choice >= len(label_words_all):
                    raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))

                label_words = label_words_all[choice]
                label_words = [label_words_per_label.strip().split(",") \
                            for label_words_per_label in label_words]
            
        elif path.endswith(".jsonl") or path.endswith(".json"):
            with open(path, "r") as f:
                label_words_all = json.load(f)
                # if it is a file containing multiple verbalizers
                if isinstance(label_words_all, list):
                    if choice >= len(label_words_all):
                        raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))
                    label_words = label_words_all[choice]
                elif isinstance(label_words_all, dict):
                    label_words = label_words_all
                    if choice>0:
                        logger.warning("Choice of verbalizer is 1, but the file  \
                        only contains one verbalizer.")
                
        self.label_words = label_words
        if self.num_classes is not None:
            num_classes = len(self.label_words)
            assert num_classes==self.num_classes, 'number of classes in the verbalizer file\
                                            does not match the predefined num_classes.'
        return self

