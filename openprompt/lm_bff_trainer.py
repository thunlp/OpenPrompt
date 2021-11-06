from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel, T5ForConditionalGeneration
from transformers.data.processors.utils import InputExample, InputFeatures

from openprompt.plms.utils import TokenizerWrapper

from .prompts import TemplateGenerator, VerbalizerGenerator
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from typing import List, Optional, Dict, Union
from . import Template, Verbalizer, PromptDataLoader
import copy
import warnings
from .trainer import ClassificationRunner
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from openprompt.utils.cuda import model_to_device
from openprompt.prompts import load_template_generator, load_verbalizer_generator
from openprompt.plms import load_plm_from_config

class LMBFFTemplate(ManualTemplate):
    """
    This is a special template used only for earch of template in LM-BFF. For example, when using T5, a template could be ``<text_a> <extra_id_0> <meta:labelword> <extra_id_1>``, where ``<meta:labelword>`` is replaced by label_words in verbalizer in `wrap_one_example` method.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        verbalizer (:obj:`ManualVerbalizer`): A verbalizer to provide label_words.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 verbalizer: ManualVerbalizer,
                 text: Optional[List[str]] = None,
                 mask_token: str = '<mask>',
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.text = text
        self.verbalizer = verbalizer
    
    def wrap_one_example(self, 
                         example: InputExample) -> List[Dict]:
        example.meta['labelword'] = self.verbalizer.label_words[example.label][0].strip()
        wrapped_example = super().wrap_one_example(example)

        # TODO: replace <mask> with special tokens in each generation model
        # e.g. in T5 multi-parts generation use <extra_id_0>, <extra_id_1>, ...
        # handle different types of plm
        current_idx = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        for d in wrapped_example[0]:
            if d['text'] == '<mask>':
                d['text'] = self.tokenizer.convert_ids_to_tokens(current_idx)
                current_idx -= 1
        return wrapped_example


def build_dataloader(dataset, template, tokenizer,tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset = dataset, 
        template = template, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=tokenizer_wrapper_class, 
        batch_size = config[split].batch_size,
        shuffle = config[split].shuffle_data,
        teacher_forcing = config[split].teacher_forcing if hasattr(config[split],'teacher_forcing') else None,
        predict_eos_token = True if config.task == "generation" else False,
        **config.dataloader
    )
    return dataloader


class LMBFFClassificationRunner:
    def __init__(self,
                train_dataset: List[InputExample],
                valid_dataset: List[InputExample],
                test_dataset: List[InputExample],
                verbalizer: Optional[Verbalizer] = None,
                template: Optional[str] = None,
                config: CfgNode = None):
        r"""
        This class implements the LM-BFF in paper (https://arxiv.org/pdf/2012.15723.pdf)
        template_text_for_auto_t: '{"placeholder":"text_a"} {"mask"} {"meta": "labelword"} {"mask"}', each {"mask"} stands for one part of template to be generated
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.model, self.tokenizer, self.model_config, self.tokenizer_wrapper = load_plm_from_config(config)
        self.auto_t = config.classification.auto_t
        self.auto_v = config.classification.auto_v
        if self.auto_t:
            print('loading auto_t model...')
            self.template_generate_model, self.template_generate_tokenizer, self.template_generate_model_config, self.template_tokenizer_wrapper = load_plm_from_config(config.template_generator)
            print('done')
        self.verbalizer = verbalizer
        self.template = template
        self.config = config
        self.check_param()

    def check_param(self):
        if self.auto_t:
            if self.verbalizer is None:
                raise ValueError("no verbalizer for template generation provided!")
            if self.template is not None:
                warnings.warn("auto_t is set True, ignore the given template")
        elif self.auto_v:
            if self.template is None:
                raise ValueError("no template for verbalizer generation provided, or set auto_t=True to automatically generate one")
            if self.verbalizer is not None:
                warnings.warn("auto_v is set True, ignore the given verbalizer")
        else:
            warnings.warn("auto_t and auto_v are both False, the trainer will degenerate to a simple classification trainer")
        
    
    def _auto_t(self, dataset, template):
        logger.info("performing auto-t...")
        if self.template_generate_model is None or self.template_generate_tokenizer is None:
            raise ValueError("template_generate_model or template_generate_tokenizer is None !")
        model = model_to_device(self.template_generate_model, self.config.environment)
        template_generator = load_template_generator(config=self.config, template_generate_model=model, tokenizer=self.template_generate_tokenizer)
        dataloader = PromptDataLoader(dataset, template, self.template_generate_tokenizer, self.template_tokenizer_wrapper, batch_size=len(dataset)) # register all data at once
        for data in dataloader:
            if self.config.environment.num_gpus > 0:
                data = data.to("cuda:{}".format(self.config.environment.local_rank))
            template_generator.register_buffer(data.input_ids, data.attention_mask, data.label) 
        template_texts = template_generator.generate() # List[str]
        template_generator.release_memory()
        del template_generator, model
        return template_texts
    
    def _auto_v(self, dataset, template):
        logger.info("performing auto-v...")
        model = copy.deepcopy(self.model)
        model = model_to_device(model, self.config.environment)
        verbalizer_generator = load_verbalizer_generator(config=self.config, model=model, tokenizer=self.tokenizer)
        if verbalizer_generator is None:
            raise ValueError("no verbalizer_generator available")
        dataloader = PromptDataLoader(dataset, template, self.tokenizer, self.tokenizer_wrapper, batch_size=self.config.test.batch_size)
        for data in dataloader:
            data = template.process_batch(data)
            if self.config.environment.num_gpus > 0:
                data = data.to("cuda:{}".format(self.config.environment.local_rank))
            verbalizer_generator.register_buffer(data)
        label_words_list = verbalizer_generator.generate() # List[List[str]]
        verbalizer_generator.release_memory()
        del verbalizer_generator, model
        return label_words_list


    def _get_best_template_text(self, template_texts_candidates, verbalizer):
        best_metrics = 0.0
        best_template_text = None
        for template_text in template_texts_candidates:
            template = ManualTemplate(self.tokenizer, template_text)
            train_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.tokenizer_wrapper, self.config, 'train')
            valid_dataloader = build_dataloader(self.valid_dataset, template, self.tokenizer, self.tokenizer_wrapper, self.config, 'dev')
            score = self._train_eval(template, verbalizer, train_dataloader, valid_dataloader)
            if score > best_metrics:
                best_metrics = score
                best_template_text = template_text
                logger.info('best template:' + str(best_template_text))
        return best_template_text
    
    def _get_best_label_words(self, verbalizer_labelwords_candidates, template, verbalizer):
        current_verbalizer = copy.deepcopy(verbalizer)
        best_metrics = 0.0
        best_label_words = None
        for label_words in verbalizer_labelwords_candidates:
            current_verbalizer.label_words = label_words
            train_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.tokenizer_wrapper, self.config, 'train')
            valid_dataloader = build_dataloader(self.valid_dataset, template, self.tokenizer, self.tokenizer_wrapper, self.config, 'dev')
            score = self._train_eval(template, current_verbalizer, train_dataloader, valid_dataloader)
            if score > best_metrics:
                best_metrics = score
                best_label_words = label_words
                logger.info('best label words:' + str(best_label_words))
        return best_label_words

    def _train_eval(self, template, verbalizer, train_dataloader, valid_dataloader):
        model = PromptForClassification(copy.deepcopy(self.model), template, verbalizer)
        runner = ClassificationRunner(model, config=self.config, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)
        runner.clean = True
        best_score = runner.fit()
        return best_score
    
    def run(self):
        '''
        if both auto_v and auto_t are set to True, perform auto_t first and then auto_v
        '''
        best_template = self.template
        best_verbalizer = self.verbalizer
        if self.auto_t:
            template_for_auto_t = LMBFFTemplate.from_config(config=self.config.template_generator.template, tokenizer=self.template_generate_tokenizer, verbalizer = self.verbalizer)
            template_texts = self._auto_t(self.train_dataset, template_for_auto_t)
            best_template_text = self._get_best_template_text(template_texts, best_verbalizer)
            best_template = ManualTemplate(self.tokenizer, best_template_text)
        if self.auto_v:
            label_words_list = self._auto_v(self.train_dataset, best_template)
            best_label_words = self._get_best_label_words(label_words_list, best_template, best_verbalizer)
            best_verbalizer.label_words = best_label_words
        
        train_dataloader = build_dataloader(self.train_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'train')
        valid_dataloader = build_dataloader(self.valid_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'dev')
        test_dataloader = build_dataloader(self.test_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'test')
        model = PromptForClassification(copy.deepcopy(self.model), best_template, best_verbalizer)
        runner = ClassificationRunner(model, config=self.config, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)
        runner.clean = False
        return runner.run()