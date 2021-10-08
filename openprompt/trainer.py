
import os
import sys

from torch.utils.data import dataloader
sys.path.append(".")

from typing import Callable, Union
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from tqdm import tqdm
import torch
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup




class ClassificationRunner(object):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        prompt_model (:obj:`Union[DataParallel, PromptForClassification]`): One ``PromptModel`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 prompt_model: Union[DataParallel, PromptForClassification],
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 config: CfgNode = None,
                 loss_function: Optional[Callable] = None,
                 ):
        self.prompt_model = prompt_model
        self.inner_model = prompt_model.module if isinstance(prompt_model, DataParallel) else prompt_model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        if loss_function is None:
            self.config_loss_function()
        else:
            self.loss_function = loss_function
        self.config_optimize()
    
    def config_loss_function(self,):
        r"""config the loss function if it's not passed.
        """
        if self.config.classification.loss_function == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            self.loss_function = torch.nn.NLLLoss()
        else:
            raise NotImplementedError
    
    def config_optimize(self,):
        r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
        
        """
        
        self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.accumulation_steps
        num_training_steps = self.train_steps_per_epoch * self.config.train.num_epochs

        if not self.config.plm.optimize.freeze_para:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.inner_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
                {'params': [p for n, p in self.inner_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
            ]

            self.model_optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.plm.optimize.lr, eps = self.config.plm.optimize.adam_eps)
            if self.config.plm.optimize.scheduler is not None:
                self.model_scheduler = get_linear_schedule_with_warmup(
                    self.model_optimizer, 
                    num_warmup_steps = self.config.plm.optimize.scheduler.num_warmup_steps, 
                    num_training_steps = num_training_steps
                )
            else:
                self.model_scheduler = None
        else:
            self.model_optimizer = None
            self.model_scheduler = None


        class Dummy:
            pass

        ## template_config 
        template_config = self.config[self.config.template]
        if hasattr(template_config, "optimize") and template_config.optimize is not None:
            if not hasattr(self.inner_model.template, "optimize"):
                # using default gradient descent optimizer.
                self.template_optimizer = AdamW(self.inner_model.template.parameters(), lr = template_config.optimize.lr)
                if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
                    self.template_scheduler = get_linear_schedule_with_warmup(
                        self.template_optimizer, 
                        num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = num_training_steps
                    )
                else:
                    self.template_scheduler = None
            else:
                self.template_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.template_optimizer, "step", self.inner_model.template.optimize)
                setattr(self.template_optimizer, "zero_grad", lambda:None)
                self.template_scheduler = None
        else:
            self.template_optimizer = None
            self.template_scheduler = None
            
        
        

        ## verbalizer_optimizer
        verbalizer_config = self.config[self.config.verbalizer]
        if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None:
            if not hasattr(self.inner_model.verbalizer, "optimize"):
                # using default gradient descent optimizer.
                self.verbalizer_optimizer = AdamW(self.inner_model.verbalizer.parameters(), lr = verbalizer_config.optimize.lr)
                if hasattr(verbalizer_config.optimize, "scheduler") and verbalizer_config.optimize.scheduler is not None:
                    self.verbalizer_scheduler = get_linear_schedule_with_warmup(
                        self.verbalizer_optimizer, 
                        num_warmup_steps = verbalizer_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = num_training_steps
                    )
                else:
                    self.verbalizer_scheduler = None
            else:
                self.verbalizer_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.verbalizer_optimizer, "step", self.inner_model.verbalizer.optimize)
                setattr(self.verbalizer_optimizer, "zero_grad", lambda:None)
                self.verbalizer_scheduler = None
        else:
            self.verbalizer_optimizer = None
            self.verbalizer_scheduler = None

        self.optimizers = [self.model_optimizer, self.template_optimizer, self.verbalizer_optimizer]
        self.schedulers = [self.model_scheduler, self.template_scheduler, self.verbalizer_scheduler]
    
    def evaluate(self, dataloader, split, post_evaluate_hook=None):
        preds = []
        labels = []
        self.prompt_model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=split):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                label = batch['label'].cpu().tolist()
                batch.pop('label')
                logits = self.prompt_model(batch)
                pred = torch.argmax(logits, dim=-1)
                preds.extend(pred.cpu().tolist())
                labels.extend(label)
        self.prompt_model.train()
        scores = {}
        scores_str = ""
        for metric in self.config.classification.metric:
            score = classification_metrics(preds, labels, metric)
            scores[metric] = score
            scores_str += "{}: {}\n".format(metric, score)
        logger.info("{} Performance: {}".format(split, scores_str))

    def train_epoch(self, epoch):
        self.prompt_model.train()
        self.prompt_model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc="Train")
        for step, batch in enumerate(pbar):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            logits = self.prompt_model(batch)
            loss = self.loss_function(logits, batch['label'])
            if self.config.train.accumulation_steps > 1:
                loss = loss / self.config.train.accumulation_steps
            sum_loss += loss.item()
            loss.backward()

            if (step+1) % self.config.train.accumulation_steps == 0:
                pbar.set_postfix({ 'loss': sum_loss })
                if self.config.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), self.config.train.max_grad_norm)
                for optimizer in self.optimizers:
                    if optimizer is not None:
                        optimizer.step()

                for scheduler in self.schedulers:
                    if scheduler is not None:
                        scheduler.step()

                for optimizer in self.optimizers:
                    if optimizer is not None:
                        optimizer.zero_grad()
                total_loss += sum_loss
                sum_loss = 0.
        logger.info("Epoch {}, avg_loss: {:.4f}".format(epoch, total_loss / self.train_steps_per_epoch))
    
    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ) and \
            not hasattr(self.inner_model.template, "optimize_to_initialize" ):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                logits = self.prompt_model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize" ):
                self.inner_model.template.optimize_to_initialize()

    def run(self):
        self.prompt_initialize()
        for epoch in range(self.config.train.num_epochs):
            self.train_epoch(epoch)
            self.evaluate(self.valid_dataloader, "Valid")
        self.evaluate(self.test_dataloader, "Test")


class GenerationRunner(object):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for generation.

    Args:
        prompt_model (:obj:`Union[DataParallel, PromptForClassification]`): One ``PromptModel`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
    """
    def __init__(self, 
                 prompt_model: Union[DataParallel, PromptForGeneration],
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 config: CfgNode = None,
                 ):
        self.prompt_model = prompt_model
        self.inner_model = prompt_model.module if isinstance(prompt_model, DataParallel) else prompt_model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.config_optimize()
    
    def config_loss_function(self,):
        r"""config the loss function if it's not passed.
        """
        if self.config.classification.loss_function == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
    
    def config_optimize(self,):
        r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
        
        """
        
        self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.accumulation_steps
        num_training_steps = self.train_steps_per_epoch * self.config.train.num_epochs

        if not self.config.plm.optimize.freeze_para:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.inner_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
                {'params': [p for n, p in self.inner_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
            ]

            self.model_optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.plm.optimize.lr)
            if self.config.plm.optimize.scheduler is not None:
                self.model_scheduler = get_linear_schedule_with_warmup(
                    self.model_optimizer, 
                    num_warmup_steps = self.config.plm.optimize.scheduler.num_warmup_steps, 
                    num_training_steps = num_training_steps
                )
            else:
                self.model_scheduler = None
        else:
            self.model_optimizer = None
            self.model_scheduler = None


        class Dummy:
            pass

        ## template_config 
        template_config = self.config[self.config.template]
        if template_config.optimize is not None:
            if not hasattr(self.inner_model.template, "optimize"):
                # using default gradient descent optimizer.
                self.template_optimizer = AdamW(self.inner_model.template.parameters(), lr = template_config.optimize.lr)
                if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
                    self.template_scheduler = get_linear_schedule_with_warmup(
                        self.template_optimizer, 
                        num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = num_training_steps
                    )
                else:
                    self.template_scheduler = None
            else:
                self.template_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.template_optimizer, "step", self.inner_model.template.optimize)
                setattr(self.template_optimizer, "zero_grad", lambda:None)
                self.verbalizer_scheduler = None
        else:
            self.template_optimizer = None
            self.template_scheduler = None
        
        self.optimizers = [self.model_optimizer, self.template_optimizer]
        self.schedulers = [self.model_scheduler, self.template_scheduler]

    def evaluate(self, dataloader, split, post_evaluate_hook=None):
        if not os.path.exists(self.config.generation.result_path):
            raise FileNotFoundError("Can't find {}".format(self.config.generation.result_path))

        # TODO: allow more flexible file name
        ret_file_name= os.path.join(self.config.generation.result_path,"{}_{}.txt".format(self.config.template, split))
        fout = open(ret_file_name,'w')
        tgt_texts = []
        generated_sentences_all = []
        logger.info("Begin generation, result written at {}".format(ret_file_name))
        for batch in tqdm(dataloader, desc=split):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            output_sequences, generated_sentences = self.inner_model.generate(batch)
            tgt_texts.extend(batch['tgt_text'])
            generated_sentences_all.extend(generated_sentences)
            for i in range(len(batch['tgt_text'])):
                fout.write("[Gold]:"+batch['tgt_text'][i]+"\n")
                fout.write("[Gen]: "+generated_sentences[i]+"\n\n")
        fout.close()
        score = generation_metric(tgt_texts, generated_sentences_all)
        logger.info("Evaluate Bleu score: {:.3f}.".format(score*100))

    def train_epoch(self, epoch):
        self.prompt_model.train()
        self.prompt_model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc="Train")
        for step, batch in enumerate(pbar):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            loss = self.prompt_model(batch).sum()  #TODO： parallel doesn't aggregate the result for some reason. to fix.
            if self.config.train.accumulation_steps > 1:
                loss = loss / self.config.train.accumulation_steps
            sum_loss += loss.item()
            loss.backward()

            if (step+1) % self.config.train.accumulation_steps == 0:
                pbar.set_postfix({ 'loss': sum_loss })
                if self.config.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), self.config.train.max_grad_norm)
                for optimizer in self.optimizers:
                    if optimizer is not None:
                        optimizer.step()

                for scheduler in self.schedulers:
                    if scheduler is not None:
                        scheduler.step()

                for optimizer in self.optimizers:
                    if optimizer is not None:
                        optimizer.zero_grad()
                total_loss += sum_loss
                sum_loss = 0.
        logger.info("Epoch {}, avg_loss: {:.4f}".format(epoch, total_loss / self.train_steps_per_epoch))
    
    def run(self):
        # currently no methods support automatic template initialization for generation
        for epoch in range(self.config.train.num_epochs):
             self.train_epoch(epoch)
             self.evaluate(self.valid_dataloader, "Valid")
        self.evaluate(self.test_dataloader, "Test")
