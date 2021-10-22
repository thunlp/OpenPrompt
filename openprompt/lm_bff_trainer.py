from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from transformers.data.processors.utils import InputExample, InputFeatures

from .prompts import TemplateGenerator, VerbalizerGenerator
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualVerbalizer, LMBFFTemplate, ManualTemplate
from typing import List, Optional, Dict, Union
from . import Template, Verbalizer, PromptDataLoader
import copy
import warnings
from .trainer import ClassificationRunner
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from openprompt.utils.cuda import model_to_device
from openprompt.prompts import load_template_generator, load_verbalizer_generator


def build_dataloader(dataset, template, tokenizer, config, split):
    dataloader = PromptDataLoader(dataset=dataset, 
                                template=template, 
                                tokenizer=tokenizer, 
                                batch_size=config[split].batch_size,
                                shuffle=config[split].shuffle_data,
                                teacher_forcing=config[split].teacher_forcing \
                                    if hasattr(config[split],'teacher_forcing') else None,
                                predict_eos_token=True if config.task=="generation" else False,
                                **config.dataloader
                                )
    return dataloader


class LMBFFClassificationRunner:
    def __init__(self,
                train_dataset: List[InputExample],
                valid_dataset: List[InputExample],
                test_dataset: List[InputExample],
                model: PreTrainedModel, 
                tokenizer: PreTrainedTokenizer,
                template_generate_tokenizer: PreTrainedTokenizer, 
                template_generate_model: PreTrainedModel,
                initial_template: Union[LMBFFTemplate, ManualTemplate],
                initial_verbalizer: ManualVerbalizer = None,
                config: CfgNode = None):
        r"""
        This class implements the LM-BFF in paper (https://arxiv.org/pdf/2012.15723.pdf)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template_generate_tokenizer = template_generate_tokenizer
        self.template_generate_model = template_generate_model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.metric = self.config.classification.metric[0]
        self.initial_template = initial_template
        self.initial_verbalizer = initial_verbalizer
        self.auto_t = config.classification.auto_t
        self.auto_v = config.classification.auto_v

        if self.auto_t:
            if not (initial_template and initial_verbalizer):
                raise ValueError("initial template and verbalizer must not be None when both auto_t and auto_v are performed")
            if not isinstance(initial_template, LMBFFTemplate):
                raise ValueError("To perform template search, initial_template must be LMBFFTemplate, however {} is provided".format(type(initial_template.__class__.__name__)))
        elif self.auto_v:
            if initial_verbalizer is not None:
                warnings.warn("only auto_v is True, the given initial_verbalizer is ignored")
            if isinstance(initial_template, LMBFFTemplate):
                raise ValueError("To perform template search, initial_template must be ManualTemplate, however LMBFFTemplate is provided")
        else:
            warnings.warn("auto_t and auto_v are both False, the trainer will degenerate to a simple classification trainer")
        
    
    def _auto_t(self, dataset, template):
        logger.info("performing auto-t...")
        if self.template_generate_model is None or self.template_generate_tokenizer is None:
            raise ValueError("template_generate_model or template_generate_tokenizer is None !")
        model = model_to_device(self.template_generate_model, self.config.environment)
        template_generator = load_template_generator(config=self.config, template_generate_model=model, tokenizer=self.template_generate_tokenizer)
        dataloader = PromptDataLoader(dataset, template, self.template_generator_tokenizer, batch_size=len(dataset)) # register all data at once
        for data in dataloader:
            if self.config.environment.num_gpus > 0:
                data = data.to("cuda:{}".format(self.config.environment.local_rank))
            template_generator.register_buffer(data.input_ids, data.attention_mask, data.label) 
        template_texts = template_generator.generate() # [['text_a', '<mask>', ...]]
        return template_texts
    
    def _auto_v(self, dataset, template):
        logger.info("performing auto-v...")
        model = copy.deepcopy(self.model)
        model = model_to_device(model, self.config.environment)
        verbalizer_generator = load_verbalizer_generator(config=self.config, model=model, tokenizer=self.tokenizer)
        if verbalizer_generator is None:
            raise ValueError("no verbalizer_generator available")
        dataloader = PromptDataLoader(dataset, template, self.tokenizer, batch_size=len(dataset))
        for data in dataloader:
            data = template.process_batch(data)
            if self.config.environment.num_gpus > 0:
                data = data.to("cuda:{}".format(self.config.environment.local_rank))
            verbalizer_generator.register_buffer(data)
        label_words_list = verbalizer_generator.generate() # List[List[str]]
        return label_words_list

    def run(self):
        '''
        if both auto_v and auto_t are set to True, perform auto_t first and then auto_v
        '''

        best_template = self.initial_template
        best_verbalizer = self.initial_verbalizer
        if self.auto_t:
            template_texts = self._auto_t(self.train_dataset, self.initial_template)
            best_template_text = self._get_best_template_text(template_texts, best_verbalizer)
            best_template = ManualTemplate(self.tokenizer, best_template_text)
        if self.auto_v:
            label_words_list = self._auto_v(self.train_dataset, best_template)
            best_label_words = self._get_best_label_words(label_words_list, best_template, best_verbalizer)
            best_verbalizer.label_words = best_label_words
        
        train_dataloader = build_dataloader(self.train_dataset, best_template, self.tokenizer, self.config, 'train')
        valid_dataloader = build_dataloader(self.train_dataset, best_template, self.tokenizer, self.config, 'dev')
        test_dataloader = build_dataloader(self.train_dataset, best_template, self.tokenizer, self.config, 'test')
        model = PromptForClassification(copy.deepcopy(self.model), best_template, best_verbalizer)
        model = model_to_device(model, self.config.environment)
        runner = ClassificationRunner(model, train_dataloader, valid_dataloader, test_dataloader, config=self.config)
        runner.run()


    def _get_best_template_text(self, template_texts_candidates, verbalizer):
        best_metrics = 0.0
        best_template_text = None
        for template_text in template_texts_candidates:
            template = ManualTemplate(self.tokenizer, template_text)
            train_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.config, 'train')
            valid_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.config, 'dev')
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
            train_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.config, 'train')
            valid_dataloader = build_dataloader(self.train_dataset, template, self.tokenizer, self.config, 'dev')
            score = self._train_eval(template, current_verbalizer, train_dataloader, valid_dataloader)
            if score > best_metrics:
                best_metrics = score
                best_label_words = label_words
                logger.info('best label words:' + str(best_label_words))
        return best_label_words

    def _train_eval(self, template, verbalizer, train_dataloader, valid_dataloader):
        model = PromptForClassification(copy.deepcopy(self.model), template, verbalizer) 
        model = model_to_device(model, self.config.environment)
        runner = ClassificationRunner(model, train_dataloader, valid_dataloader, config=self.config)
        best_score = 0.0
        for epoch in range(self.config.train.num_epochs):
            runner.train_epoch(epoch)
            scores = runner.evaluate(valid_dataloader, 'Valid')
            score = scores[self.metric]
            if score > best_score:
                best_score = score
        return best_score