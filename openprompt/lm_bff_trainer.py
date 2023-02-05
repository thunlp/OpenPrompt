from transformers.data.processors.utils import InputExample, InputFeatures


from openprompt import PromptDataLoader, PromptForClassification
from openprompt.pipeline_base import PromptModel

from openprompt.prompts import ManualVerbalizer, ManualTemplate
from typing import List, Optional, Dict, Union
from . import Verbalizer, PromptDataLoader
import copy
import warnings
from .trainer import ClassificationRunner
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from openprompt.utils.cuda import model_to_device
from openprompt.prompts import load_template_generator, load_verbalizer_generator
from openprompt.plms import load_plm_from_config

class ManualTemplateWithoutParse(ManualTemplate):
    def on_text_set(self):
        pass


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
    r"""
        This runner implements the LM-BFF training process in paper `Making Pre-trained Language Models Better Few-shot Learners(Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_.

        Args:
            train_dataset (:obj:`List[InputExample]`): The dataset for training
            valid_dataset (:obj:`List[InputExample]`): The dataset for validation
            test_dataset (:obj:`List[InputExample]`): The dataset for test
            verbalizer (:obj:`Optional[Verbalizer]`): The manually designed verbalizer for template generation. Defaults to None.
            template (:obj:`Optional[Verbalizer]`): The manually designed template for verbalizer generation. Defaults to None.
            config (:obj:`CfgNode`): A configuration object
        """
    def __init__(self,
                train_dataset: List[InputExample],
                valid_dataset: List[InputExample],
                test_dataset: List[InputExample],
                verbalizer: Optional[Verbalizer] = None,
                template: Optional[str] = None,
                config: CfgNode = None):

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.model, self.tokenizer, self.model_config, self.tokenizer_wrapper = load_plm_from_config(config)
        self.auto_t = config.classification.auto_t
        self.auto_v = config.classification.auto_v

        self.verbalizer = verbalizer
        self.template = template
        self.config = config
        self._check_param()

    def _check_param(self):
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


    def _auto_t(self):
        logger.info("performing auto-t...")
        template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm_from_config(self.config.template_generator)
        model = model_to_device(template_generate_model, self.config.environment)
        template_generator = load_template_generator(config=self.config, model = model, tokenizer=template_generate_tokenizer, tokenizer_wrapper = template_tokenizer_wrapper, verbalizer = self.verbalizer)
        template_texts = template_generator.generate(self.train_dataset) # List[str]
        template_generator.release_memory()
        del template_generator, model
        return template_texts

    def _auto_v(self, template):
        logger.info("performing auto-v...")
        model = copy.deepcopy(self.model)
        model = model_to_device(model, self.config.environment)
        verbalizer_generator = load_verbalizer_generator(config=self.config, model=model, tokenizer=self.tokenizer)
        dataloader = PromptDataLoader(self.train_dataset, template, self.tokenizer, self.tokenizer_wrapper, batch_size=self.config.test.batch_size)
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
            template = ManualTemplateWithoutParse(self.tokenizer, template_text)
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
        r"""
        Run LM-BFF. if both `auto_v` and `auto_v` are set to True in ``config``, automatic template generation will be performed first.
        """
        best_template = self.template
        best_verbalizer = self.verbalizer
        if self.auto_t:
            template_texts = self._auto_t()
            best_template_text = self._get_best_template_text(template_texts, best_verbalizer)
            best_template = ManualTemplateWithoutParse(self.tokenizer, best_template_text)
        if self.auto_v:
            label_words_list = self._auto_v(best_template)
            best_label_words = self._get_best_label_words(label_words_list, best_template, best_verbalizer)
            best_verbalizer.label_words = best_label_words

        train_dataloader = build_dataloader(self.train_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'train')
        valid_dataloader = build_dataloader(self.valid_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'dev')
        test_dataloader = build_dataloader(self.test_dataset, best_template, self.tokenizer, self.tokenizer_wrapper, self.config, 'test')
        model = PromptForClassification(copy.deepcopy(self.model), best_template, best_verbalizer)
        runner = ClassificationRunner(model, config=self.config, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)
        runner.clean = False
        return runner.run()