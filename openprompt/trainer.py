import os
import sys
sys.path.append(".")

from torch.utils.data import dataloader


from openprompt.utils.utils import load_checkpoint, save_checkpoint
from typing import Callable, OrderedDict, Union
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from tqdm import tqdm
import torch
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup




class BaseRunner(object):
    r"""A base runner for training without training tricks.
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
                 ):
        self.prompt_model = prompt_model
        self.inner_model = prompt_model.module if isinstance(prompt_model, DataParallel) else prompt_model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.config_optimize()
    
    def config_loss_function(self,):
        raise NotImplementedError
    
    def config_optimize(self,):
        raise NotImplementedError
 
    def evaluate(self, dataloader, split, post_evaluate_hook=None):
        raise NotImplementedError

    def train_epoch(self, epoch):
        raise NotImplementedError
    
    def prompt_initialize(self):
        r"""Some initialization works
        """
        pass

    def run(self, start_epoch: int=0, max_score: float=0.0):
        if start_epoch == 0:
            self.prompt_initialize()
            max_score = None
        for epoch in range(start_epoch, self.config.train.num_epochs):
            total_loss = self.train_epoch(epoch)
            scores = self.evaluate(self.valid_dataloader, "Valid")
            model_state_dict = self.inner_model.state_dict()
            if self.config.plm.optimize.freeze_para:
                model_state_dict.pop('plm')
            state_dict = {
                "epoch": epoch+1,
                "state_dict": self.inner_model.state_dict(),
                "optimizer": [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in self.optimizers] ,
                "scheduler": [sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in self.schedulers],
                "scores": scores,
                "max_score": max_score
            }
            cur_score = scores.popitem()[1]

            is_best = ((cur_score - max_score)>=0) == \
                self.config.checkpoint.higher_better if max_score is not None else True
            if is_best:
                max_score = cur_score
            save_checkpoint(state_dict = state_dict, 
                            is_best=(is_best and self.config.checkpoint.save_best), 
                            save_path=self.config.logging.path)
        state_dict = load_checkpoint(load_path=self.config.logging.path,
                        load_best = self.config.checkpoint.save_best,
                        map_location="cpu", # cpu to prevent CUDA out of memory.
                        )
        self.inner_model.load_state_dict(state_dict['state_dict'])
        self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        self.evaluate(self.test_dataloader, "Test")

    def resume(self, ):
        logger.info("Resume Training ...")
        try:
            state_dict = load_checkpoint(load_path=self.config.logging.path,
                    load_best = False,
                    map_location="cpu", # cpu to prevent CUDA out of memory.
                    )
        except FileNotFoundError:
            logger.warning("No checkpoint found in {}, start from scratch.".format(self.config.logging.path))
            self.run()
            return 
        
        # load state to model
        self.inner_model.load_state_dict(state_dict['state_dict'])
        self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        # load state to optimizers
        for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(op_state)
        for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
            if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                scheduler.load_state_dict(sc_state)
        # run
        self.run(start_epoch=state_dict['epoch'], max_score=state_dict['max_score'])
        
    def test(self, ):
        logger.info("Resume Training and direct test...")
        try:
            state_dict = load_checkpoint(load_path=self.config.logging.path,
                    load_best = False,
                    map_location="cpu", # cpu to prevent CUDA out of memory.
                    )
        except FileNotFoundError:
            logger.error("No checkpoint found in {}, can't test.".format(self.config.logging.path))
            exit()
        
        # load state to model
        self.inner_model.load_state_dict(state_dict['state_dict'])
        self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        self.evaluate(self.test_dataloader, "Test")




class ClassificationRunner(BaseRunner):
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
        super().__init__(prompt_model=prompt_model,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         config=config)

        if loss_function is None:
            self.config_loss_function()
        else:
            self.loss_function = loss_function
    
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
        
        self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.gradient_accumulation_steps
        num_training_steps = self.train_steps_per_epoch * self.config.train.num_epochs

        if not self.config.plm.optimize.freeze_para:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.inner_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
                {'params': [p for n, p in self.inner_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
            ]

            self.model_optimizer = AdamW(
                optimizer_grouped_parameters,
                lr = self.config.plm.optimize.lr,
                betas = self.config.plm.optimize.betas,
                eps = self.config.plm.optimize.eps
            )
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
        scores = OrderedDict()
        scores_str = ""
        for metric in self.config.classification.metric:
            score = classification_metrics(preds, labels, metric)
            scores[metric] = score
            scores_str += "{}: {}\n".format(metric, score)
        logger.info("{} Performance: {}".format(split, scores_str.strip()))
        return scores

    def train_epoch(self, epoch):
        self.prompt_model.train()
        self.prompt_model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc="Train epoch {}".format(epoch))
        for step, batch in enumerate(pbar):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            logits = self.prompt_model(batch)
            loss = self.loss_function(logits, batch['label'])
            if self.config.train.gradient_accumulation_steps > 1:
                loss = loss / self.config.train.gradient_accumulation_steps
            sum_loss += loss.item()
            loss.backward()
            if (step+1) % self.config.train.gradient_accumulation_steps == 0:
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
        logger.info("Epoch {}, avg_loss: {:.4f}, total_loss: {:.4f}".format(epoch, total_loss / self.train_steps_per_epoch, total_loss))
        return total_loss
    
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


class GenerationRunner(BaseRunner):
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
        super().__init__(prompt_model=prompt_model,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         config=config)
    
    def config_loss_function(self,):
        r""" No need to config loss_function in generation.
        """
        pass
    
    def config_optimize(self,):
        r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
        
        """
        
        self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.gradient_accumulation_steps
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
                no_decay = template_config.optimize.no_decay
                weight_decay = template_config.optimize.weight_decay
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in self.inner_model.template.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],'weight_decay': weight_decay},
                    {'params': [p for n, p in self.inner_model.template.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],'weight_decay': 0.0}
                ]

                self.template_optimizer = AdamW(self.inner_model.template.parameters(), 
                                                lr = template_config.optimize.lr,
                                                betas = template_config.optimize.betas,
                                                eps = template_config.optimize.eps)
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
        ret_file_name= os.path.join(self.config.logging.path,"{}_generated_text.txt".format(split))
        
        tgt_texts = []
        generated_sentences_all = []
        for batch in tqdm(dataloader, desc=split):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            output_sequences, generated_sentences = self.inner_model.generate(batch, **self.config.generation)
            tgt_texts.extend(batch['tgt_text'])
            generated_sentences_all.extend(generated_sentences)
            
        fout = open(ret_file_name,'w')
        for i in range(len(generated_sentences_all)):
            fout.write(generated_sentences_all[i]+"\n")
        fout.close()

        scores = OrderedDict()
        scores_str = ""
        for metric in self.config.generation.metric:
            score = generation_metric(generated_sentences_all, tgt_texts, metric)
            scores[metric] = score
            scores_str += "{}: {}\n".format(metric, score)
        logger.info("{} Performance: {}".format(split, scores_str.strip()))
        return scores

    def train_epoch(self, epoch):
        self.prompt_model.train()
        self.prompt_model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc="Train epoch {}".format(epoch))
        for step, batch in enumerate(pbar):
            batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            loss = self.prompt_model(batch).mean()  #TODO：unbanlanced batch chunks
            if self.config.train.gradient_accumulation_steps > 1:
                loss = loss / self.config.train.gradient_accumulation_steps
            sum_loss += loss.item()
            loss.backward()

            if (step+1) % self.config.train.gradient_accumulation_steps == 0:
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
        logger.info("Epoch {}, avg_loss: {:.4f}, total_loss: {:.4f}".format(epoch, total_loss / self.train_steps_per_epoch, total_loss))
        return total_loss
