
from openprompt.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer

class PtuningTemplate(ManualTemplate):
    """
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        prompt_encoder_type (:obj:`str`): head above the embedding layer of new tokens. Can be ``lstm`` or ``mlp``.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        soft_token (:obj:`str`, optional): The special token for new token. Default to ``<soft>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["soft_token_ids", "loss_ids", 'shortenable_ids']

    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 prompt_encoder_type: str = "lstm",
                 text:  Optional[List[str]] = None,
                 mask_token: str = '<mask>',
                 soft_token: str = '<soft>',
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.prompt_encoder_type = prompt_encoder_type
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.soft_token = soft_token
        self.text = text

    def get_default_soft_token_ids(self) -> List[int]:
        r"""get the new token indices for the template
        e.g. when self.text is ['<text_a>', '<soft>', '<soft>', '<mask>', '.'],
        output is [0, 1, 2, 0, 0]
        """
        # TODO ptuing supervised use same new token for each <soft> ?
        idx = []
        num_soft_token = 0
        for token in self.text:
            if token == self.soft_token:
                num_soft_token += 1
                idx.append(num_soft_token)
            else:
                idx.append(0)
        return idx

    def on_text_set(self):
        r"""
        when template text was set, generate parameters needed in p-tuning input embedding phrase
        """
        self.text = self.parse_text(self.text)
        self.num_soft_token = sum([token == self.soft_token for token in self.text])
        self.generate_parameters()

    def generate_parameters(self) -> None:
        r"""
        generate parameters needed for new tokens' embedding in P-tuning
        """
        if self.num_soft_token == 0:
            return
        self.new_embedding = nn.Embedding(self.num_soft_token, self.embedding_size)
        self.new_ids = nn.Parameter(torch.LongTensor(list(range(self.num_soft_token))), requires_grad = False)
        if self.prompt_encoder_type == "lstm":
            self.new_lstm_head = nn.LSTM(
                input_size = self.embedding_size,
                hidden_size = self.embedding_size, # TODO P-tuning different in LAMA & FewGLUE
                # TODO dropout different in LAMA and FewGLUE
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(2*self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "mlp":
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        else:
            raise ValueError("unknown prompt_enocder_type")
            
    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for new tokens, use a brand new embedding layer, with MLP or LSTM head
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])

        if self.num_soft_token != 0:
            new_embeds = self.new_embedding(self.new_ids).unsqueeze(0)
            if self.prompt_encoder_type == "lstm":
                new_embeds = self.new_lstm_head(new_embeds)[0]
            new_embeds = self.new_mlp_head(new_embeds)

            replace_idxs = torch.nonzero(batch['soft_token_ids']>0).view(-1, self.num_soft_token, 2)
            for b in range(replace_idxs.shape[0]):
                for i in range(self.num_soft_token):
                    inputs_embeds[b][replace_idxs[b][i][1]] = new_embeds[0][i]

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
