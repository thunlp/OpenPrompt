
import os
import string
from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template

import torch
from torch import nn
from sys import exit

class MixedTemplate(Template):
    r"""The Mixed Template class defined by a string of `text`. See more examples in the `tutorial <https://github.com/thunlp/OpenPrompt/blob/ca27491101df0108a8dd753e5b1e79bf591f65d3/tutorial/1.1_mixed_template.py>`_.

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
    """
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):

        super().__init__(tokenizer = tokenizer, placeholder_mapping = placeholder_mapping)

        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.text = text

    def get_default_soft_token_ids(self) -> List[int]:
        return self.soft_token_ids

    def prepare(self):
        r"""get the soft token indices ( soft_token_ids ) for the template

        ``"soft_id"`` can be used to reference the previous soft token, which means these tokens use the same embeddings.
        **Note that ``"soft_id"`` should have index start from 1 but not 0**

        e.g. when self.text is ``'{"soft": None} {"soft": "the", "soft_id": 1} {"soft": None} {"soft": "it", "soft_id": 3} {"soft_id": 1} {"soft": "was"} {"mask"}'``,
        output is [1, 2, 3, 4, 2, 5, 0]
        """
        num_soft_token = 0
        text = []
        soft_token_ids = []
        idx_mp = {}
        emb_mp = {}
        for d in self.text:
            if "soft" not in d and "soft_id" not in d:
                text.append(d)
                soft_token_ids.append(0)
                continue

            old_num = num_soft_token

            if "soft_id" in d:
                if not isinstance(d["soft_id"], int) or d["soft_id"] <= 0:
                    raise ValueError(f'soft_id should be integer greater than zero, but get {d["soft_id"]}')
                if d["soft_id"] in idx_mp:
                    id_list = idx_mp[d["soft_id"]]
                    text.extend([{"soft":None} for _ in range(len(id_list))])
                    soft_token_ids.extend(id_list)
                    continue
                else:
                    if "soft" not in d: d["soft"] = None

            if d["soft"] is None:
                if "duplicate" in d:
                    if "same" in d and d["same"]:
                        num_soft_token += 1
                        id_list = [num_soft_token for _ in range(d["duplicate"])]
                    else:
                        num_soft_token += d["duplicate"]
                        id_list = list(range(old_num+1, num_soft_token+1))
                else:
                    num_soft_token += 1
                    id_list = [num_soft_token]
                text.extend([{"soft":""} for _ in range(len(id_list))])
            else:
                token_ids = self.tokenizer(d["add_prefix_space"] + d["soft"], add_special_tokens=False)["input_ids"]
                surface_forms = self.tokenizer.convert_ids_to_tokens(token_ids)
                assert len(token_ids) == len(surface_forms)
                num_soft_token += len(token_ids)
                id_list = list(range(old_num+1, num_soft_token+1))
                for idx, soft_id in enumerate(id_list):
                    emb_mp[soft_id] = token_ids[idx]

                text.extend([{"soft": surface_form} for surface_form in surface_forms])
            soft_token_ids.extend(id_list)

            if "soft_id" in d:
                idx_mp[d["soft_id"]] = id_list

        self.num_soft_token = num_soft_token
        self.text = text
        self.soft_token_ids = soft_token_ids

        # Generate the embedding needed for soft tokens

        self.soft_embedding = nn.Embedding(1 + self.num_soft_token, self.embedding_size)
        for soft_id, token_id in emb_mp.items():
            self.soft_embedding.weight.data[soft_id, :] = self.raw_embedding.weight.data[token_id, :].clone().detach().requires_grad_(True)

        # if "post_processing" in d:
        #     if d["post_processing"] == "mlp":
        #         pass # TODO one mlp or more than one
        #     else:
        #         raise ValueError(f'post_processing of {d["post_processing"]} is not supported yet')

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


    def on_text_set(self):
        """
        when template text was set

        1. parse text

        2. generate parameter needed
        """

        self.text = self.parse_text(self.text)
        self.prepare()


    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        raw_embeds = self.raw_embedding(batch['input_ids'])
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds, raw_embeds)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
