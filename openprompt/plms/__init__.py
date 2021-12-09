from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from .seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from .lm import LMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2LMHeadModel      
from collections import namedtuple
from yacs.config import CfgNode

from openprompt.utils.logging import logger

    
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5TokenizerWrapper
    }),
    't5-lm':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5LMTokenizerWrapper,
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]


def load_plm(model_name, model_path, specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
    
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`model_config`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name: # add pad token for gpt 
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    return model, tokenizer, model_config, wrapper

def load_plm_from_config(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
    
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`model_config`: The wrapper class of this plm.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    # you can change huggingface model_config here
    # if 't5'  in plm_config.model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in plm_config.model_name: # add pad token for gpt
        if "<pad>" not in config.plm.specials_to_add:
            config.plm.specials_to_add.append("<pad>")
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)
    wrapper = model_class.wrapper
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config, wrapper

def add_special_tokens(model: PreTrainedModel, 
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer. 

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer



