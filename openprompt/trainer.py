import os
import sys
sys.path.append(".")

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.utils.cuda import model_to_device

import dill
from typing import Callable, Union
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
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
        prompt_model (:obj:`nn.Module`): One ``nn.Module`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 model: PromptForClassification,
                 config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                ):
        self.model = model_to_device(model, config.environment)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.configure_optimizers()

        self.cur_epoch = 0
        self.best_score = None
        self.global_step = 0

    @property
    def num_training_steps(self) -> int:
        """Total training steps"""
        batches = len(self.train_dataloader)

        effective_accum = self.trainer.num_gpus * self.config.train.gradient_accumulation_steps
        return (batches // effective_accum) * self.config.num_epochs
        
    @property
    def steps_per_epoch(self) -> int:
        """num of training steps per epoch"""
        return self.num_training_steps // self.trainer.num_epochs

    @property
    def inner_model(self):
        return self.model.module if isinstance(self.model, DataParallel) else self.prompt_model
    
    def configure_optimizers(self):
        r"""config the optimizer and scheduler for
        
        1. model
        
        2. template
        
        3. verbalizer(optional)
        """
        
        self.optimizers, self.schedulers = self.inner_model.configure_optimizers() # TODO

    def load_checkpoint(self, ckpt: str, load_state = True) -> bool:
        logger.info(f"Loading Checkpoint {self.checkpoint_path(ckpt)}...")
        try:
            state_dict = torch.load(
                self.checkpoint_path(ckpt),
                pickle_module = dill,
                map_location = "cpu"
            )
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found")
            return False
        
        # load state to model
        self.model = self.inner_model
        self.model.load_state_dict(state_dict['state_dict'])
        self.model = model_to_device(self.model, self.config.environment)

        if load_state:
            # load state to optimizers
            for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.load_state_dict(op_state)
            for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
                if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                    scheduler.load_state_dict(sc_state)

            # load training state
            self.cur_epoch = state_dict['cur_epoch']
            self.best_score = state_dict['best_score']
            self.global_step = state_dict['global_step']
        logger.info(f"Load Checkpoint finished, the current validation metrics: {state_dict['validation_metrics']}")
        return True

    def save_checkpoint(self, ckpt:str, save_state = True, **kwargs):
        logger.info("Saving checkpoint ...")
        state_dict = {
            "state_dict": self.inner_model.state_dict(),
        }
        state_dict.update(kwargs)

        if save_state:
            state_dict.update({
                "optimizer": [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in self.optimizers] ,
                "scheduler": [sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in self.schedulers],

                "cur_epoch": self.cur_epoch,
                "best_score": self.best_score,
                "global_step": self.global_step,
            })
        torch.save(state_dict, self.checkpoint_path(ckpt), pickle_module = dill)

    def logger(self):
        pass # TODO

    def save_results(self, split, **kwargs):
        for name, values in kwargs.items():
            file_name = os.path.join(self.config.logging.path, f"{split}_{name}.txt")
            with open(file_name, 'w') as fout:
                for value in values:
                    print(value, file = fout)
 
    def inference_epoch(self, split: str): 
        outputs = []
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split=='validation' else self.test_dataloader
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()

                outputs.extend( self.inference_step(batch, batch_idx) )

        metrics = self.inference_epoch_end(split, outputs)
        logger.info(f"{split} Performance: {metrics}")

    def training_epoch(self, epoch):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc = "Train epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to("cuda:{}".format(self.config.environme.format(epoch, total_loss / self.train_steps_per_epoch, total_loss))nt.local_rank)).to_dict()

            loss = self.training_step(batch, batch_idx)

            if self.config.train.gradient_accumulation_steps > 1:
                loss = loss / self.config.train.gradient_accumulation_steps
            sum_loss += loss.item()
            loss.backward()
            if (batch_idx+1) % self.config.train.gradient_accumulation_steps == 0:
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
                self.global_step += 1
        logger.info(f"Training epoch {epoch}, avg_loss: {total_loss/self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
    
    def on_fit_start(self):
        """Some initialization works"""
        pass

    def fit(self, ckpt: Optional[str] = None):
        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")
        if self.cur_epoch == 0:
            self.on_fit_start()
        for self.cur_epoch in range(self.cur_epoch, self.config.train.num_epochs):
            self.training_epoch(self.cur_epoch)
            metrics = self.inference_epoch("validation")
            score = metrics.popitem(last=False)[1] # TODO the first metric is the most important one
            self.save_checkpoint('last', {"validation_metrics": metrics})
            if self.best_score is None or ((score - self.max_score) >= 0) == self.config.checkpoint.higher_better:
                self.save_checkpoint('best', {"validation_metrics": metrics})
                self.best_score = score

    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            if not self.load_checkpoint('best', load_state = False):
                logger.error("Test cannot be performed")
                exit()
        return self.inference_epoch("test")

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        return self.test(ckpt = 'best')
        

class ClassificationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 model: PromptForClassification,
                 config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 loss_function: Optional[Callable] = None,
                 ):
        super().__init__(model = model,
                         config = config,
                         train_dataloader = train_dataloader,
                         valid_dataloader = valid_dataloader,
                         test_dataloader = test_dataloader,
                        )
        self.loss_function = loss_function if loss_function else self.configure_loss_function()
    
    def config_loss_function(self,):
        r"""config the loss function if it's not passed."""
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError
    
    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def inference_epoch_end(self, split, outputs):
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)

        self.save_results(split, {
            'preds': preds,
            'labels': labels,
        })

        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            metric = classification_metrics(preds, labels, metric_name)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss
    
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
                logits = self.model(batch)
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
        model (:obj:`PromptForGeneration`): One ``PromptForGeneration`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
    """
    def __init__(self, 
                 model: PromptForGeneration,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 config: CfgNode = None,
                ):
        super().__init__(model=model,
                         config=config,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                        )

    def inference_step(self, batch, batch_idx):
        target = batch['tgt_text'] # TODO pop?
        _, pred = self.model.generate(batch, **self.config.generation)
        return pred, target # these are already a cpu list

    def inference_epoch_end(self, split, outputs):
        preds = []
        targets = []
        for pred, target in outputs:
            preds.extend(pred)
            targets.extend(target)

        self.save_results(split, {
            'preds': preds,
            'targets': targets
        })

        metrics = OrderedDict()
        for metric_name in self.config.generation.metric:
            metric = generation_metric(preds, targets, metric_name)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        return loss
