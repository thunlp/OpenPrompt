
from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from openprompt.data_utils.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template, Verbalizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer

class PrefixTuningTemplate(Template):
    r"""This template different from most template in that this emplate doesn't need to 
    wrap the input sentences with the template. The new tokens are always prepended to
    the language model. A mapping is used to map the new_tokens embeddings in to the
    past_key_value, and then input into the language model. The mask token of this 
    template is automatically the last token.

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained model.
        plm_config (:obj:`PretrainedConfig`): The configuration of the current pre-trained model.
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model.
        mapping_hook (:obj:`nn.Module`, optional):
        text (:obj:`str`, optional): 
        mask_token (:obj:`str`, optional):
        num_token (:obj:`int`, optional):
        placeholder_mapping (:obj:`dict`):
        prefix_dropout (:obj:`float`, optional): The dropout rate for the prefix sequence.
    """
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self, 
                 model: PreTrainedModel,
                 plm_config: PretrainedConfig,
                 tokenizer: PreTrainedTokenizer,
                 mapping_hook: Optional[nn.Module] = None,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_token: Optional[int] = 5,
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                 prefix_dropout: Optional[float] = 0.0,
                ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.mapping_hook = mapping_hook
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.num_token = num_token
        self.config = plm_config

        if isinstance(self.config, T5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
        elif isinstance(self.config, GPT2Config):
            self.n_layer = self.config.n_layer
            self.n_embd = self.config.n_embd
            self.n_head = self.config.n_head
        self.mid_dim = self.n_embd  # TODO to be modified

        self.match_n_layer = self.n_layer
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.default_text1 = "<text_a> <eos> <mask> <eos>".split()
        self.default_text2 = "<text_a> <text_b> <eos> <mask> <eos>".split()

        self.text = text
        
        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

        self.plm_modified = False # flag to indicate whether the function of plm are replaced for prefix tuning.
    
    def on_text_set(self):
        pass

    def get_past_key_values(self, batch_size):
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.config.is_encoder_decoder:
            decoder_input_tokens = self.decoder_input_tokens.unsqueeze(0).expand(batch_size, -1)
            decoder_temp_control = self.decoder_wte(decoder_input_tokens)
            decoder_past_key_values = self.decoder_control_trans(decoder_temp_control) #bsz, seqlen, layer*emb
            _, decoder_seqlen, _ = decoder_past_key_values.shape
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, self.match_n_decoder_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

            return (past_key_values, decoder_past_key_values)
        
        return (past_key_values,)

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
        self.wte = nn.Embedding(self.num_token, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd))

        if self.config.is_encoder_decoder:
            self.decoder_input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
            self.decoder_wte = nn.Embedding(self.num_token, self.n_embd)
            self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd))


    def wrap_one_example(self, example) -> List[Dict]:
        if self.text is None:
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal token, use the embedding inside PLM
        for new token, use MLP or LSTM
        """
        batch_size = batch['input_ids'].size(0)
        past_key_values = self.get_past_key_values(batch_size)
        self.past_key_values = past_key_values
        return batch

    
    def modify_plm(self, model):
        if self.plm_modified:
            return None
        if isinstance(model, T5ForConditionalGeneration):
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(model.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = self.past_key_values[0][layer_id]
                    if kwargs['attention_mask'] is not None:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = torch.cat([torch.ones((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                    return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)

            backup_decoder_self_attn_forward_functions = []
            backup_decoder_cross_attn_forward_functions = []
            for i, layer_module in enumerate(model.decoder.block):
                backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                def modified_decoder_self_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = self.past_key_values[1][layer_id]
                    if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                        pass
                    elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1) +self.num_token:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = torch.cat([torch.ones((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                    else:
                        raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                             attention mask length {}".format(kwargs['past_key_value'][0].size(-2), 
                             args[0].size(-2),kwargs['attention_mask'].size(-1)))
                    return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)

                backup_decoder_cross_attn_forward_functions.append(layer_module.layer[1].forward)
                def modified_decoder_cross_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    # kwargs['past_key_value'] = None #self.past_key_values[1][layer_id]
                    return backup_decoder_cross_attn_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[1].forward = partial(modified_decoder_cross_attn_forward, layer_id=i)
            
            self.backup_encoder_forward_functions = backup_encoder_forward_functions
            self.backup_decoder_self_attn_forward_functions = backup_decoder_self_attn_forward_functions
            self.backup_decoder_cross_attn_forward_functions = backup_decoder_cross_attn_forward_functions

        elif isinstance(model, GPT2LMHeadModel):
            backup_block_forward_functions = []
            for i, layer_module in enumerate(model.transformer.h):
                backup_block_forward_functions.append(layer_module.attn.forward)
                def modified_block_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs['layer_past'] is None:
                        kwargs['layer_past'] = self.past_key_values[0][layer_id]
                    am = kwargs['attention_mask']
                    kwargs['attention_mask'] = torch.cat([torch.ones((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                    if isinstance(kwargs['layer_past'], tuple):
                        assert kwargs['layer_past'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1), "Size not match {} + {} != {}".format(kwargs['layer_past'].size(), args[0].size(-2), kwargs['attention_mask'].size())
                    else:
                        assert kwargs['layer_past'].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1), "Size not match {} + {} != {}".format(kwargs['layer_past'].size(), args[0].size(-2),  kwargs['attention_mask'].size())
                    return backup_block_forward_functions[layer_id](*args, **kwargs)
                layer_module.attn.forward = partial(modified_block_forward, layer_id=i)
            self.backup_block_forward_functions = backup_block_forward_functions
        else:
            raise NotImplementedError
        self.plm_modified = True

    def retrieve_plm(self, model):
        if not self.plm_modified:
            return None
        if isinstance(model, T5ForConditionalGeneration):
            for i, layer_module in enumerate(model.encoder.block):
                layer_module.layer[0].forward = self.backup_encoder_forward_functions[i]
            for i, layer_module in enumerate(model.decoder.block):
                layer_module.layer[0].forward = self.backup_decoder_self_attn_forward_functions[i]
                layer_module.layer[1].forward = self.backup_decoder_cross_attn_forward_functions[i]
        elif isinstance(model, GPT2LMHeadModel):
            for i, layer_module in enumerate(model.transformer.h):
                layer_module.attn.forward = self.backup_block_forward_functions[i]
        else:
            raise NotImplementedError
        self.plm_modified = False

