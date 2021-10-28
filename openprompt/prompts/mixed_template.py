
import os
from openprompt.utils.logging import logger



from openprompt.data_utils.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class MixedTemplate(Template):
    """Mixed of manual token, trainable token and trainable that initialized with given hard token

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mixed_token_start (:obj:`str`, optional): The special mark for the start of the trainable token. Default to ``{``
        mixed_token_end (:obj:`str`, optional): The special mark for the end of the trainable token. Default to ``}``
    """
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 mixed_token_start: Optional[str] = '{',
                 mixed_token_end: Optional[str] = '}',
                ):

        self.tokenizer = self.tokenizer

        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.text = text

    def get_default_loss_ids(self):
        '''Get the loss indices for the template using mask.
        e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask": True}.'``,
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
        special tokens are not shortenable. As far as we are concerned, almost every template will use this default setting. 
        e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask": True}.'``,
        output is ``[1, 0, 0, 0, 0, 0]``.
        
        Returns:
            :obj:`List[int]`: A list of integers in the range ``[0, 1]``:

            - 1 for the input tokens.
            - 0 for the template sequence tokens.
        """
        return [1 if 'placeholder' in d else 0 for d in self.text]

    def get_default_soft_token_ids(self) -> List[int]:
        return self.soft_token_ids
    
    def prepare(self):
        r"""get the trainable token indices for the template
        
        ``"soft_id"`` can be used to reference the previous soft token, which means these tokens use the same embeddings.
        **Note that ``"soft_id"`` should have index start from 1 but not 0**

        e.g. when self.text is ``'{"soft": None} {"soft": "the", "soft_id": 1} {"soft": None} {"soft": "it", "soft_id": 3} {"soft_id": 1} {"soft": "was"} {"mask": True}'``,
        output is [1, 2, 3, 4, 2, 5, 0]

        TODO
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

            if "soft_id" in d:
                if not isinstance(d["soft_id"], int) or d["soft_id"] <= 0:
                    raise ValueError(f'soft_id should be integer greater than zero, but get {d["soft_id"]}')
                if d["soft_id"] in idx_mp:
                    previous_ids = idx_mp[d["soft_id"]]
                    text.extend([{"soft"} for _ in range(len(id_list))])
                    soft_token_ids.extend(previous_ids)
                    continue
                else:
                    if "soft" not in d: d["soft"] = None
                    old_num = num_soft_token

            if d["soft"] is None:
                num_soft_token += 1
            else:
                token_ids = self.tokenizer(d["add_prefix"] + d["soft"], add_special_tokens=False)["input_ids"]
                if len(token_ids) > 1 and d.get("single_token", "True"):
                    logger.warning(f"""
                    soft prompt's hard prompt {d["soft"]} tokenize to more than one tokens: {self.tokenizer.convert_ids_to_tokens(token_ids)}
                    By default we use the first token {self.tokenizer.convert_ids_to_tokens(token_ids)[0]}.
                    You can use {{"soft": "complicated", "single_token": False}} to support multiple tokens
                    """)
                    token_ids = token_ids[:1]
                num_soft_token += len(token_ids)
                for idx, soft_id in enumerate(list(range(old_num+1, num_soft_token+1))):
                    emb_mp[soft_id] = token_ids[idx]

            id_list = list(range(old_num+1, num_soft_token+1))
            text.extend([{"soft"} for _ in range(len(id_list))])
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

        if "post_processing" in d:
            if d["post_processing"] == "mlp":
                pass # TODO one mlp or more than one
            else:
                raise ValueError(f'post_processing of {d["post_processing"]} is not supported yet')

    def parse_text(self, text: str) -> List[Dict]:
        if not isinstance(text, str):
            raise ValueError("text of mixed_template should be str type")

        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix": ' ' if (i > 0 and text[i-1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix"] = ' '
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j]
                i = j

            else:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        break
                    j = j + 1
                if j == len(text):
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{'+text[i+1:j]+'}'
                try:
                    d.update(eval(dict_str))
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

    def incorporate_text_example(self,
                                 example: InputExample
                                ):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d.get("post_processing", lambda x:x)(
                    getattr(example, self.placeholder_mapping[d['placeholder']])
                )
            elif 'meta' in d:
                text[i] = d.get("post_processing", lambda x:x)(
                    getattr(example.meta, d['meta'])
                )
            elif 'soft' in d:
                text[i] = ''; # unused
            elif 'text' in d:
                text[i] = d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

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
            text = fin.readlines()[choice]
        self.text = text
        return self
