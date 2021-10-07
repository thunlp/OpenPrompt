from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from .seq2seq import T5TokenizerWrapper
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

    
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertForMaskedLM,
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
    }),
}

TOKENIZER_WRAPPER_MAPPING = {
    BertTokenizer: MLMTokenizerWrapper,
    RobertaTokenizer: MLMTokenizerWrapper,
    AlbertTokenizer: MLMTokenizerWrapper,
    OpenAIGPTTokenizer: LMTokenizerWrapper,
    GPT2Tokenizer: LMTokenizerWrapper,
    T5Tokenizer: T5TokenizerWrapper,
}

def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def get_tokenizer_wrapper(tokenizer: PreTrainedTokenizer) -> TokenizerWrapper:
    try:
        wrapper_class = TOKENIZER_WRAPPER_MAPPING[type(tokenizer)]
    except KeyError:
        logger.info("tokenizer type not in TOKENIZER_WRAPPER_MAPPING")
    return wrapper_class


def load_plm(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
    
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    # you can change huggingface model_config here
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config

def add_special_tokens(model: PreTrainedModel, 
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = ['<pad>']):
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



