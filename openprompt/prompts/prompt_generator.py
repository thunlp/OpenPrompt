from abc import abstractmethod
from builtins import ValueError
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from ..utils import logger
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForMaskedLM, RobertaForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from tqdm import tqdm
from typing import List, Optional, Dict
import itertools
import numpy as np
from ..utils import signature
from ..config import convert_cfg_to_dict
from torch.nn.parallel import DataParallel


class TemplateGenerator:
    r""" Automatic Template Search from LM-BFF

    Args:
        beam_width: beam search width
        max_length: maximum length of generated template
        length_limit: length limit for each part of content
        target_number: number of parts to generate, e.g. in T5, every <extra_id_{}> token is one part
    """
    def __init__(self, 
                template_generate_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None, 
                 forbidden_word_ids: Optional[List[int]] = []):
        self.template_generate_model = template_generate_model
        self.tokenizer = tokenizer
        self.target_number = target_number # number of parts to generate in one sample
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_limit = length_limit
        self.probs_buffer, self.labels_buffer = None, None

        # Forbid single space token, "....", and "..........", and some other tokens based on vocab
        self.forbidden_word_ids = forbidden_word_ids
        self.sent_end_id = self.tokenizer.convert_tokens_to_ids('.')

        self.input_ids_buffer, self.attention_mask_buffer, self.labels_buffer = None, None, None

    def register_buffer(self, input_ids, attention_mask, labels):
        if self.input_ids_buffer is None :
            self.input_ids_buffer = input_ids.detach()
            self.attention_mask_buffer = attention_mask.detach()
            self.labels_buffer = labels.detach()
        else:
            self.input_ids_buffer = torch.vstack([self.input_ids_buffer, input_ids.detach()])
            self.attention_mask_buffer = torch.vstack([self.attention_mask_buffer, attention_mask.detach()])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels.detach()])

    @abstractmethod
    def get_next_part_token_id(self, part_id: int) -> int:
        r"""get the start token id for next part
        """
        raise NotImplementedError
    
    def convert_template(self, text_list: List[str]) -> List[str]:
        r"""convert the generated template into a standard template for downstream prompt model, return a list of str
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_templates(self):
        inner_model = self.template_generate_model.module if isinstance(self.template_generate_model, DataParallel) else self.template_generate_model
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
                        aggr_output.append(self.template_generate_model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.to(input_ids.device)[start:end])[0])
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

        self.templates_text = []
        for item in current_output:
            generate_text = []
            for i in item['output']:
                generate_text.append(self.tokenizer._convert_id_to_token(i))
            self.templates_text.append(' '.join(self.convert_template(generate_text)))
    
    def _show_template(self):
        logger.info("Templates are \n{}".format('\n'.join(self.templates_text)))

    def generate(self):
        self.template_generate_model.eval()
        with torch.no_grad():
            self.get_templates()
            self._show_template()
        return self.templates_text

    @classmethod
    def from_config(cls, config, **kwargs,):
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        template_generator = cls(**init_dict)
        return template_generator
    
    def release_memory(self):
        self.template_generate_model = self.template_generate_model.cpu()
        

class T5TemplateGenerator(TemplateGenerator): # TODO merge it into Base class
    r""" Automatic Template Search from LM-BFF using T5
    """
    def __init__(self, 
                 template_generate_model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None,
                 forbidden_word_ids: Optional[List[int]] = [3, 19794, 22354]):
        super().__init__(template_generate_model = template_generate_model,
                        tokenizer = tokenizer,
                        max_length = max_length,
                        target_number= target_number,
                        beam_width = beam_width,
                        length_limit = length_limit,
                        forbidden_word_ids = forbidden_word_ids)

    def get_part_token_id(self, part_id):
        return self.tokenizer.convert_tokens_to_ids('<extra_id_0>') - part_id

    def convert_template(self, generate_text_list):
        text_list = self.tokenizer.convert_tokens_to_string(generate_text_list).replace('<extra_id_0>', '{"placeholder":"text_a"}').replace('<extra_id_1>', ' {"mask"}').replace('<extra_id_2>', ' {"placeholder": "text_b"}').replace('</s>', '').replace('  ', ' ').split(' ')
        # incase no <extra_id_1> (generation stop by maximum length)
        if '{"mask"}' not in text_list:
            text_list.append('{"mask"}')
        if '{"placeholder": "text_b"}' not in text_list:
            text_list.append('{"placeholder": "text_b"}')
        return text_list


class VerbalizerGenerator:
    r""" Automatic Label Words Search from https://arxiv.org/pdf/2012.15723.pdf

    Args:
        candidate_num: the number of candidates for further selection
        label_word_num_per_class: candidate label words per class
        probs_buffer: stores the probability $q_{P,t}(1|\mathbf{x})$ to be 
                      used in later label words selection.
        label_buffer: stores the label $y$ to be used in later label words
                      selection.
    """
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 candidate_num: int,
                 label_word_num_per_class: Optional[int] = 1):
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
    
    def post_process(self, word):
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return word.lstrip('Ġ')
        elif isinstance(inner_model, BertForMaskedLM):
            return word
        else:
            raise NotImplementedError("not implemented for {}".format(type(inner_model))) # TODO add more model
        
    def invalid_label_word(self, word):
        '''
        make sure label word is the proper start of a word
        '''
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return (not word.startswith('Ġ'))
        elif isinstance(inner_model, BertForMaskedLM):
            return False
        else:
            raise NotImplementedError("not implemented for {}".format(type(inner_model))) # TODO

    def generate(self):
        self.label_words_ids = self._find_verbalizer()
        self.label_words = [[self.post_process(word) for word in self.tokenizer.convert_ids_to_tokens(i)] for i in self.label_words_ids]
        self._show_verbalizer()
        return self.label_words
            
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
    def from_config(cls, config, **kwargs,):
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer_generator = cls(**init_dict)
        return verbalizer_generator
    
    def release_memory(self):
        self.model = self.model.cpu()