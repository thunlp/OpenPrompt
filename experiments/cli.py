import os
import sys
sys.path.append(".")


from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.prompts import T5TemplateGenerator, VerbalizerGenerator
from typing import Union
from torch.nn.parallel.data_parallel import DataParallel
from re import template
from torch._C import device
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from tqdm import tqdm
import argparse
import torch
from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import get_model_class
from openprompt import PromptDataLoader, PromptModel
from openprompt.prompts import load_template, load_verbalizer, load_template_generator, load_verbalizer_generator
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.utils.metrics import classification_metrics
from openprompt.utils.calibrate import calibrate
from transformers import  AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from openprompt.config import get_yaml_config
from openprompt.plms import load_plm
from openprompt.data_utils import load_dataset
from openprompt.utils.cuda import model_to_device
from openprompt.utils.utils import check_config_conflicts
import logging





def get_config():
    parser = argparse.ArgumentParser("classification config")
    parser.add_argument("--config_yaml", type=str, help='the configuration file for this experiment.')
    parser.add_argument("--resume", action="store_true", help='whether to resume a training from the latest checkpoint.\
           It will fall back to run from initialization if no lastest checkpoint are found.')
    parser.add_argument("--test", action="store_true", help='whether to resume a training from the latest checkpoint.\
           It will fall back to run from initialization if no lastest checkpoint are found.') #
    args = parser.parse_args()
    config = get_yaml_config(args.config_yaml)
    check_config_conflicts(config)
    logger.info("CONFIGS:\n{}\n{}\n".format(config, "="*40))
    return config, args


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

def save_config_to_yaml(config):
    from contextlib import redirect_stdout
    saved_yaml_path = os.path.join(config.logging.path, "config.yaml")
    with open(saved_yaml_path, 'w') as f:
        with redirect_stdout(f): print(config.dump())
    logger.info("Config saved as {}".format(saved_yaml_path))


def main():
    config, args = get_config()
    # init logger, create log dir and set log level, etc.
    if not args.resume:
        config.logging.path_base = os.path.join(sys.path[0], config.logging.path_base)
        EXP_PATH = config_experiment_dir(config)
    else:
        EXP_PATH = config.logging.path
    
    init_logger(EXP_PATH+"/log.txt", config.logging.file_level, config.logging.console_level)
    # save config to the logger directory
    if not args.resume:
        save_config_to_yaml(config)
    # set seed
    set_seed(config)
    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config = load_plm(config)
    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(config)
    
    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            template_generate_model, template_generate_tokenizer, template_generate_config = load_plm(config.template_generator)
            template_generate_model = model_to_device(template_generate_model, config.environment)
            verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
            template = load_template(config=config, model=template_generate_model, tokenizer=template_generate_tokenizer, plm_config=template_generate_config, verbalizer=verbalizer)
            template_generator = load_template_generator(config=config, template_generate_model=template_generate_model, tokenizer=template_generate_tokenizer)
            verbalizer_generator = load_verbalizer_generator(config=config, model=plm_model, tokenizer=plm_tokenizer)
        else:
            # define prompt
            template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
            verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
            # load prompt’s pipeline model
        prompt_model = PromptForClassification(plm_model, template, verbalizer)
            
    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(plm_model, template, gen_config=config.generation)

    # move the model to device:
    prompt_model = model_to_device(prompt_model, config.environment)

    # process data and get data_loader
    if config.learning_setting == 'full':
        pass
    elif config.learning_setting == 'few_shot':
        if config.few_shot.few_shot_sampling is not None:
            sampler = FewShotSampler(
                num_examples_per_label = config.sampling_from_train.num_examples_per_label,
                also_sample_dev = config.sampling_from_train.also_sample_dev,
                num_examples_per_label_dev = config.sampling_from_train.num_examples_per_label_dev
            )
            train_dataset, valid_dataset = sampler(
                train_dataset = train_dataset,
                valid_dataset = valid_dataset,
                seed = config.sampling_from_train.seed
            )
    elif config.learning_setting == 'zero_shot':
        pass
    
    if config.calibrate is not None:
        assert isinstance(prompt_model, PromptForClassification), "The type of model doesn't support calibration."
        calibrate(prompt_model, config)

    train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, config, "train")
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, config, "dev")
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, config, "test")
    # test_dataloader = valid_dataloader  # if the test size is big, replace it with valid_dataloader for debugging.
    # test_dataset = valid_dataset
    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            runner = LMBFFClassificationRunner(train_dataset = train_dataset, 
                                        valid_dataset = valid_dataset, 
                                        test_dataset = test_dataset, 
                                        model= plm_model, 
                                        tokenizer = plm_tokenizer, 
                                        template_generator_tokenizer = template_generate_tokenizer,
                                        initial_template = template,
                                        initial_verbalizer = verbalizer,
                                        template_generator = template_generator,
                                        verbalizer_generator = verbalizer_generator,
                                        config = config)
        else:
            runner = ClassificationRunner(prompt_model = prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    config = config)
    elif config.task == "generation":
        runner = GenerationRunner(prompt_model = prompt_model,
                                train_dataloader = train_dataloader,
                                valid_dataloader = valid_dataloader,
                                test_dataloader = test_dataloader,
                                config = config)
        
    else:
        raise NotImplementedError
    if not args.resume:
        runner.run()
    else:
        if args.test: #
            runner.test()#
        else:#
            runner.resume()




if __name__ == "__main__":
    main()
    # get_config()
