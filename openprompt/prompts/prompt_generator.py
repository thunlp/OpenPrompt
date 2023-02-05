from abc import abstractmethod
from builtins import ValueError
from typing import List, Optional, Dict, Union
from tokenizers import Tokenizer
import json
import torch
import torch.nn.functional as F
from yacs.config import CfgNode
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.pipeline_base import PromptDataLoader, PromptModel

from openprompt.prompt_base import Template, Verbalizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from ..utils import logger
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Optional, Dict
import itertools
import numpy as np
from ..utils import signature
from ..config import convert_cfg_to_dict
from torch.nn.parallel import DataParallel

class LMBFFTemplateGenerationTemplate(ManualTemplate):
    """
    This is a special template used only for search of template in LM-BFF. For example, a template could be ``{"placeholder": "text_a"}{"mask"}{"meta":"labelword"}{"mask"}``, where ``{"meta":"labelword"}`` is replaced by label_words in verbalizer in `wrap_one_example` method, and ``{"mask"}`` is replaced by special tokens used for generation, for T5, it is ``<extra_id_0>, <extra_id_1>, ...``.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        verbalizer (:obj:`ManualVerbalizer`): A verbalizer to provide label_words.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    def __init__(self,
                 tokenizer: T5Tokenizer,
                 verbalizer: ManualVerbalizer,
                 text: Optional[List[str]] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         text = text,
                         placeholder_mapping=placeholder_mapping)
        self.verbalizer = verbalizer

    def wrap_one_example(self,
                         example: InputExample) -> List[Dict]:
        example.meta['labelword'] = self.verbalizer.label_words[example.label][0].strip()
        wrapped_example = super().wrap_one_example(example)
        return wrapped_example

class TemplateGenerator:
    r""" This is the automatic template search implementation for `LM-BFF <https://arxiv.org/pdf/2012.15723.pdf>`_. It uses a generation model to generate multi-part text to fill in the template. By jointly considering all samples in the dataset, it uses beam search decoding method to generate a designated number of templates with the highest probability. The generated template may be uniformly used for all samples in the dataset.

    Args:
        model (:obj:`PretrainedModel`): A pretrained model for generation.
        tokenizer (:obj:`PretrainedTokenizer`): A corresponding type tokenizer.
        tokenizer_wrapper (:obj:`TokenizerWrapper`): A corresponding type tokenizer wrapper class.
        max_length (:obj:`Optional[int]`): The maximum length of total generated template. Defaults to 20.
        target_number (:obj:`Optional[int]`): The number of separate parts to generate, e.g. in T5, every <extra_id_{}> token stands for one part. Defaults to 2.
        beam_width (:obj:`Optional[int]`): The beam search width.  Defaults to 100.
        length_limit (:obj:`Optional[List[int]]`): The length limit for each part of content, if None, there is no limit. If not None, the list should have a length equal to `target_number`. Defaults to None.
        forbidden_word_ids (:obj:`Optional[List[int]]`): Any tokenizer-specific token_id you want to prevent from generating. Defaults to `[]`, i.e. all tokens in the vocabulary are allowed in the generated template.
    """
    def __init__(self,
                model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper: Tokenizer,
                 verbalizer: Verbalizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None,
                 forbidden_word_ids: Optional[List[int]] = [],
                 config: CfgNode = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.verbalizer= verbalizer
        self.target_number = target_number # number of parts to generate in one sample
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_limit = length_limit
        self.probs_buffer, self.labels_buffer = None, None

        # Forbid single space token, "....", and "..........", and some other tokens based on vocab
        self.forbidden_word_ids = forbidden_word_ids
        self.sent_end_id = self.tokenizer.convert_tokens_to_ids('.')

        self.input_ids_buffer, self.attention_mask_buffer, self.labels_buffer = None, None, None

        self.config = config

    @property
    def device(self):
        r"""
        return the device of the model
        """
        if isinstance(self.model, DataParallel):
            return self.model.module.device
        else:
            return self.model.device

    def _register_buffer(self, data):
        if self.input_ids_buffer is None :
            self.input_ids_buffer = data.input_ids.detach()
            self.attention_mask_buffer = data.attention_mask.detach()
            self.labels_buffer = data.label.detach()
        else:
            self.input_ids_buffer = torch.vstack([self.input_ids_buffer, data.input_ids.detach()])
            self.attention_mask_buffer = torch.vstack([self.attention_mask_buffer, data.attention_mask.detach()])
            self.labels_buffer = torch.hstack([self.labels_buffer, data.label.detach()])

    @abstractmethod
    def get_part_token_id(self, part_id: int) -> int:
        r"""
        Get the start token id for the current part. It should be specified according to the specific model type. For T5 model, for example, the start token for `part_id=0` is `<extra_id_0>`, this method should return the corresponding token_id.
        Args:
            part_id (:obj:`int`): The current part id (starts with 0).
        Returns:
            token_id (:obj:`int`): The corresponding start token_id.
        """
        raise NotImplementedError

    def convert_template(self, generated_template: List[str], original_template: List[Dict]) -> List[Dict]:
        r"""
        Given original template used for template generation,convert the generated template into a standard template for downstream prompt model, return a ``str``
        Example:
        generated_template: ['<extra_id_0>', 'it', 'is', '<extra_id_1>', 'one', '</s>']
        original_template: [{'add_prefix_space': '', 'placeholder': 'text_a'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': ' ', 'meta': 'labelword'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': '', 'text': '.'}]
        return: [{'add_prefix_space': '', 'placeholder': 'text_a'}, {'add_prefix_space': ' ', 'text': 'it'}, {'add_prefix_space': ' ', 'text': 'is'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': ' ', 'text': 'one'}, {'add_prefix_space': '', 'text': '.'}]
        """
        i = 0
        part_id = 0
        while generated_template[i] != self.tokenizer.additional_special_tokens[part_id] and i < len(generated_template) - 1:
            i += 1
        assert generated_template[i] == self.tokenizer.additional_special_tokens[part_id], print('invalid generated_template {}, missing token {}'.format(generated_template, self.tokenizer.additional_special_tokens[part_id]))
        i += 1

        output = []
        for d in original_template:
            if 'mask' in d:
                j = i + 1
                part_id += 1
                while generated_template[j] != self.tokenizer.additional_special_tokens[part_id] and j < len(generated_template) - 1:
                    j += 1
                
                text = self.tokenizer.convert_tokens_to_string(generated_template[i:j]).split(" ")
                output.append({"text": text[0], "add_prefix_space": d.get("add_prefix_space", "")})
                for t in text[1:]:
                    output.append({"text": t, "add_prefix_space": d.get("add_prefix_space", "")})
                i = j + 1
            elif 'meta' in d and d['meta'] == 'labelword':
                d_new = {}
                d_new["mask"] = None
                d_new["add_prefix_space"] = d["add_prefix_space"]
                output.append(d_new)
            else:
                output.append(d)
        return output


    def _get_templates(self):
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        input_ids = self.input_ids_buffer
        attention_mask = self.attention_mask_buffer

        ori_decoder_input_ids = torch.zeros((input_ids.size(0), self.max_length)).long()
        ori_decoder_input_ids[..., 0] = inner_model.config.decoder_start_token_id


        # decoder_input_ids: decoder inputs for next regressive generation
        # ll: log likelihood
        # output_id: which part of generated contents we are at
        # output: generated content so far
        # last_length (deprecated): how long we have generated for this part
        current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
        for i in tqdm(range(self.max_length - 2)):
            new_current_output = []
            for item in current_output:
                if item['output_id'] > self.target_number:
                    # Enough contents
                    new_current_output.append(item)
                    continue
                decoder_input_ids = item['decoder_input_ids']

                # Forward
                batch_size = 32
                turn = input_ids.size(0) // batch_size
                if input_ids.size(0) % batch_size != 0:
                    turn += 1
                aggr_output = []
                for t in range(turn):
                    start = t * batch_size
                    end = min((t + 1) * batch_size, input_ids.size(0))

                    with torch.no_grad():
                        aggr_output.append(self.model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.to(input_ids.device)[start:end])[0])
                aggr_output = torch.cat(aggr_output, 0)

                # Gather results across all input sentences, and sort generated tokens by log likelihood
                aggr_output = aggr_output.mean(0)
                log_denominator = torch.logsumexp(aggr_output[i], -1).item()
                ids = list(range(inner_model.config.vocab_size))
                ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
                ids = ids[:self.beam_width+3]

                for word_id in ids:
                    output_id = item['output_id']

                    if word_id == self.get_part_token_id(output_id) or word_id == self.tokenizer.eos_token_id:
                        # Finish one part
                        if self.length_limit is not None and item['last_length'] < self.length_limit[output_id - 1]:
                            check = False
                        else:
                            check = True
                        output_id += 1
                        last_length = 0
                    else:
                        last_length = item['last_length'] + 1
                        check = True

                    output_text = item['output'] + [word_id]
                    ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                    new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                    new_decoder_input_ids[:] = decoder_input_ids
                    new_decoder_input_ids[..., i + 1] = word_id

                    if word_id in self.forbidden_word_ids:
                        check = False

                    # Forbid continuous "."
                    if len(output_text) > 1 and output_text[-2] == self.sent_end_id and output_text[-1] == self.sent_end_id:
                        check = False

                    if check:
                        # Add new results to beam search pool
                        new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                        new_current_output.append(new_item)

            if len(new_current_output) == 0:
                break

            new_current_output.sort(key=lambda x: x['ll'], reverse=True)
            new_current_output = new_current_output[:self.beam_width]
            current_output = new_current_output

        return [self.tokenizer.convert_ids_to_tokens(item['output']) for item in current_output]

    def _show_template(self):
        logger.info("Templates are \n{}".format('\n'.join(self.templates_text)))


    @classmethod
    def from_config(cls, config: CfgNode, **kwargs,):
        r"""
        Returns:
            template_generator (:obj:`TemplateGenerator`)
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        init_dict['config'] = config
        template_generator = cls(**init_dict)
        return template_generator

    def release_memory(self):
        self.model = self.model.cpu()

    def generate(self, dataset: List[InputExample]):
        r"""
        Args:
            dataset (:obj:`List[InputExample]`): The dataset based on which template it to be generated.
        Returns:
            template_text (:obj:`List[str]`): The generated template text
        """
        template_for_auto_t = LMBFFTemplateGenerationTemplate.from_config(config=self.config.template, tokenizer=self.tokenizer, verbalizer = self.verbalizer)

        dataloader = PromptDataLoader(dataset, template_for_auto_t, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.tokenizer_wrapper, batch_size=len(dataset), decoder_max_length=128) # register all data at once
        for data in dataloader:
            data = data.to(self.device)
            self._register_buffer(data)

        self.model.eval()
        with torch.no_grad():
            self.templates_text = self._get_templates() # List[str]
            original_template = template_for_auto_t.text
            self.templates_text = [self.convert_template(template_text, original_template) for template_text in self.templates_text]
            self._show_template()
        return self.templates_text

class T5TemplateGenerator(TemplateGenerator):
    r"""
    Automatic template search using T5 model. This class inherits from ``TemplateGenerator``.
    """
    def __init__(self,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 tokenizer_wrapper: Tokenizer,
                 verbalizer: Verbalizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None,
                 forbidden_word_ids: Optional[List[int]] = [3, 19794, 22354],
                 config: CfgNode = None):
        super().__init__(model = model,
                        tokenizer = tokenizer,
                        tokenizer_wrapper=tokenizer_wrapper,
                        verbalizer = verbalizer,
                        max_length = max_length,
                        target_number= target_number,
                        beam_width = beam_width,
                        length_limit = length_limit,
                        forbidden_word_ids = forbidden_word_ids,
                        config=config)

    def get_part_token_id(self, part_id):
        return self.tokenizer.additional_special_tokens_ids[part_id]

    # def convert_template(self, generate_text_list):
    #     # original_template = self.template_for_auto_t.text
    #     text_list = self.tokenizer.convert_tokens_to_string(generate_text_list).replace('<extra_id_0>', '{"placeholder":"text_a"}').replace('<extra_id_1>', ' {"mask"}').replace('<extra_id_2>', ' {"placeholder":"text_b"}').replace('</s>', '').replace('  ', ' ').split(' ')
    #     # in case no <extra_id_1> (generation stop by maximum length)
    #     if '{"mask"}' not in text_list:
    #         text_list.append('{"mask"}')
    #     if '{"placeholder":"text_b"}' not in text_list:
    #         text_list.append('{"placeholder":"text_b"}')
    #     return text_list


class VerbalizerGenerator:
    r"""
    This is the automatic label word search implementation in `LM-BFF <https://arxiv.org/pdf/2012.15723.pdf>`_.

    Args:
        model (:obj:`PretrainedModel`): A pre-trained model for label word generation.
        tokenizer (:obj:`PretrainedTokenizer`): The corresponding tokenize.
        candidate_num (:obj:`Optional[int]`): The number of label word combinations to generate. Validation will then be performed on each combination. Defaults to 100.
        label_word_num_per_class (:obj:`Optional[int]`): The number of candidate label words per class. Defaults to 100.
    """
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 candidate_num: Optional[int] = 100,
                 label_word_num_per_class: Optional[int] = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_num = candidate_num
        self.label_word_num_per_class = label_word_num_per_class
        self.probs_buffer, self.labels_buffer = None, None

    def register_buffer(self, data):
        self.model.eval()
        with torch.no_grad():
            inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
            forward_keys = signature(inner_model.forward).args
            input_batch = {key: data[key] for key in data if key in forward_keys}
            logits = self.model.forward(**input_batch).logits[data['loss_ids']==1]
        logits = F.softmax(logits.detach(),dim=-1)
        if self.probs_buffer is None:
            self.probs_buffer = logits
            self.labels_buffer = data.label.detach()
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, data.label.detach()])

    @abstractmethod
    def post_process(self, word: str):
        r"""
        Post-processing for generated labrl word.

        Args:
            word (:obj:`str`): The original word token.

        Returns:
            processed_word (:obj:`str`): The post-processed token.
        """
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return word.lstrip('Ġ')
        elif isinstance(inner_model, BertForMaskedLM):
            return word
        else:
            raise RuntimeError("{} is not supported yet".format(type(inner_model))) # TODO add more model

    @abstractmethod
    def invalid_label_word(self, word: str):
        r"""
        Decide whether the generated token is a valid label word. Heuristic strategy can be implemented here, e.g. requiring that a label word must be the start token of a word.

        Args:
            word (:obj:`str`): The token.
        Returns:
            is_invalid (:obj:`bool`): `True` if it cannot be a label word.
        """
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return (not word.startswith('Ġ'))
        elif isinstance(inner_model, BertForMaskedLM):
            return False
        else:
            raise RuntimeError("{} is not supported yet".format(type(inner_model))) # TODO

    def _show_verbalizer(self):
        logger.info("Verbalizer is {}".format(self.label_words))


    def _find_verbalizer(self):
        logger.info("Finding verbalizer ...")
        label_words =  self._get_top_words()
        label_words = self._get_top_group(candidates=label_words)
        return label_words

    def _eval_group(self, group):
        label_logits = self.probs_buffer[:,torch.tensor(group)]
        preds = torch.argmax(label_logits, axis=-1)
        correct = torch.sum(preds == self.labels_buffer)
        return (correct / len(self.labels_buffer)).item()

    def _get_top_group(self, candidates: List[List[int]]):
        groups = list(itertools.product(*candidates))
        group_scores = list(map(self._eval_group, groups))

        # Take top-n.
        best_idx = np.argsort(-np.array(group_scores))[:self.candidate_num]
        best_groups = [groups[i] for i in best_idx]
        return best_groups

    def _get_top_words(self):
        label_words_ids = []
        for label_id in torch.unique(self.labels_buffer):
            scores = self.probs_buffer[self.labels_buffer==label_id].mean(axis=0).cpu().numpy()
            kept = []
            for i in np.argsort(-scores):
                word = self.tokenizer.convert_ids_to_tokens([i])[0]
                if self.invalid_label_word(word):
                    continue
                kept.append(i)
            label_words_ids.append(kept[:self.label_word_num_per_class])
        return label_words_ids

    @classmethod
    def from_config(cls, config: CfgNode, **kwargs,):
        r"""
        Returns:
            verbalizer_generator (:obj:`VerbalizerGenerator`)
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer_generator = cls(**init_dict)
        return verbalizer_generator

    def release_memory(self):
        self.model = self.model.cpu()

    def generate(self):
        r"""
        Generate label words.

        Returns:
            label_words (:obj:`List[List[str]]`): A list of generated label word.
        """

        self.label_words_ids = self._find_verbalizer()
        self.label_words = [[self.post_process(word) for word in self.tokenizer.convert_ids_to_tokens(i)] for i in self.label_words_ids]
        self._show_verbalizer()
        return self.label_words

class RobertaVerbalizerGenerator(VerbalizerGenerator):
    def __init__(self,
                 model: RobertaForMaskedLM,
                 tokenizer: RobertaTokenizer,
                 candidate_num: Optional[int] = 100,
                 label_word_num_per_class: Optional[int] = 100):
        super().__init__(
                        model = model,
                        tokenizer = tokenizer,
                        candidate_num = candidate_num,
                        label_word_num_per_class = label_word_num_per_class)

    def invalid_label_word(self, word: str):
        return (not word.startswith('Ġ'))

    def post_process(self, word: str):
        return word.lstrip('Ġ')