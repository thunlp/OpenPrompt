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



class LMBFFClassificationRunner:
    def __init__(self,
                train_dataset: List[InputExample],
                valid_dataset: List[InputExample],
                test_dataset: List[InputExample],
                model: PreTrainedModel, 
                tokenizer: PreTrainedTokenizer,
                template_generator_tokenizer: PreTrainedTokenizer, 
                initial_template: Union[LMBFFTemplate, ManualTemplate],
                initial_verbalizer: ManualVerbalizer = None,
                template_generator: TemplateGenerator = None,
                verbalizer_generator: VerbalizerGenerator = None,
                config: CfgNode = None):
        r"""
        This class implements the LM-BFF in paper (https://arxiv.org/pdf/2012.15723.pdf)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template_generator_tokenizer = template_generator_tokenizer
        self.max_epoch = config.classification.generation_epoch
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.template_generator = template_generator
        self.verbalizer_generator = verbalizer_generator
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
        
    
    def _auto_t(self, dataset, template, verbalizer):
        logger.info("performing auto-t...")
        if self.template_generator is None:
            raise ValueError("no template_generator available")
        dataloader = PromptDataLoader(dataset, template, self.template_generator_tokenizer, batch_size=len(dataset)) # register all data at once
        for data in dataloader:
            data = data.to("cuda:{}".format(self.config.environment.local_rank))
            self.template_generator.register_buffer(data.input_ids, data.attention_mask, data.label) 
        template_texts = self.template_generator.generate() # [['text_a', '<mask>', ...]]
        best_template_text = self._get_best_template_text(template_texts, verbalizer)
        return best_template_text
    
    def _auto_v(self, dataset, template, verbalizer, batch_size=32):
        logger.info("performing auto-v...")
        if self.verbalizer_generator is None:
            raise ValueError("no verbalizer_generator available")
        dataloader = PromptDataLoader(dataset, template, self.tokenizer, batch_size=batch_size)
        for data in dataloader:
            data = template.process_batch(data)
            data = data.to("cuda:{}".format(self.config.environment.local_rank))
            self.verbalizer_generator.register_buffer(data)
        label_words_list = self.verbalizer_generator.generate() # List[List[str]]
        best_label_words = self._get_best_label_words(label_words_list, template, verbalizer)
        return best_label_words

    def run(self):
        '''
        if both auto_v and auto_t are set to True, perform auto_t first and then auto_v
        '''

        best_template = self.initial_template
        best_verbalizer = self.initial_verbalizer
        if self.auto_t:
            best_template_text = self._auto_t(self.train_dataset, self.initial_template, best_verbalizer)
            best_template = ManualTemplate(self.tokenizer, best_template_text)
        if self.auto_v:
            best_label_words = self._auto_v(self.train_dataset, best_template, best_verbalizer)
            best_verbalizer.label_words = best_label_words
        
        train_dataloader = PromptDataLoader(self.train_dataset, best_template, self.tokenizer)
        valid_dataloader = PromptDataLoader(self.valid_dataset, best_template, self.tokenizer)
        test_dataloader = PromptDataLoader(self.test_dataset, best_template, self.tokenizer)
        model = PromptForClassification(copy.deepcopy(self.model), best_template, best_verbalizer)
        model = model_to_device(model, self.config.environment)
        runner = ClassificationRunner(model, train_dataloader, valid_dataloader, test_dataloader, config=self.config)
        runner.run()


    def _get_best_template_text(self, template_texts_candidates, verbalizer):
        best_metrics = 0.0
        best_template_text = None
        for template_text in template_texts_candidates:
            print(template_text)
            template = ManualTemplate(self.tokenizer, template_text)
            train_dataloader = PromptDataLoader(self.train_dataset, template, self.tokenizer)
            valid_dataloader = PromptDataLoader(self.valid_dataset, template, self.tokenizer)
            score = self._train_eval(template, verbalizer, train_dataloader, valid_dataloader)
            if score > best_metrics:
                best_template_text = template_text
        return best_template_text
    
    def _get_best_label_words(self, verbalizer_labelwords_candidates, template, verbalizer):
        current_verbalizer = copy.deepcopy(verbalizer)
        best_metrics = 0.0
        best_label_words = None
        for label_words in verbalizer_labelwords_candidates:
            current_verbalizer.label_words = label_words
            train_dataloader = PromptDataLoader(self.train_dataset, template, self.tokenizer)
            valid_dataloader = PromptDataLoader(self.valid_dataset, template, self.tokenizer)
            score = self._train_eval(template, current_verbalizer, train_dataloader, valid_dataloader)
            if score > best_metrics:
                best_label_words = label_words
        return best_label_words

    def _train_eval(self, template, verbalizer, train_dataloader, valid_dataloader):
        model = PromptForClassification(copy.deepcopy(self.model), template, verbalizer) 
        model = model_to_device(model, self.config.environment)
        runner = ClassificationRunner(model, train_dataloader, valid_dataloader, config=self.config)
        best_score = 0.0
        for epoch in range(self.max_epoch):
            runner.train_epoch(epoch)
            scores = runner.evaluate(valid_dataloader, 'Valid')
            score = scores[self.metric]
            if score > best_score:
                best_score = score
        return best_score