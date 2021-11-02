
from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from openprompt.data_utils import InputFeatures
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
    template is automatically the last token. Currently, our implementation of 
    prefix_tuning doens't support adding past_key_values to the encoder side of an 
    encoder_decoder architecture such as T5 without modifying the T5 source code. 
    (T5's source code doens't support adding past_key_values to the encoder in their code base. )

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
                 tokenizer: PreTrainedTokenizer,
                 mapping_hook: Optional[nn.Module] = None,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_token: Optional[int] = 5,
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                 prefix_dropout: Optional[float] = 0.0,
                 mid_dim: Optional[int] =  512,
                ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        raw_embedding = model.get_input_embeddings()
        self.config = model.config
        self.mapping_hook = mapping_hook
        self.embedding_size = raw_embedding.weight.shape[-1]
        self.num_token = num_token

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
        self.mid_dim = mid_dim

        self.match_n_layer = self.n_layer
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.default_text1 = '{"placeholder": "text_a"} {"mask"}'  #TODO whether <eos> should be added here?
        self.default_text2 = '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}'

        self.text = text
        
        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

        self.plm_modified = False # flag to indicate whether the function of plm are replaced for prefix tuning.
    

    def on_text_set(self):
        self.text = self.parse_text(self.text)
        self.generate_parameters()


    def get_past_key_values(self, batch_size):
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
        self.wte = nn.Embedding(self.num_token, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd))


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
        # self.past_key_values = past_key_values
        if 'attention_mask' in batch:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
        batch['past_key_values'] = past_key_values
        return batch
