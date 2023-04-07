# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation,
# Which we do not include in it.

import argparse
import torch
from openprompt import plms
from openprompt.plms import *
from transformers import GPTJConfig, GPTJModel, GPTJForCausalLM, GPT2Tokenizer
plms._MODEL_CLASSES["gptj"]= ModelClass(**{"config": GPTJConfig, "tokenizer": GPT2Tokenizer, "model": GPTJForCausalLM,
"wrapper": LMTokenizerWrapper})
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from openprompt.data_utils.conditional_generation_dataset import UltraChatProcessor
from transformers import AdamW
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from transformers.optimization import get_linear_schedule_with_warmup
from accelerate import Accelerator
from torchmetrics import MeanMetric
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from accelerate.utils import set_seed




def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log

def evaluate(args, model, val_dataloader, accelerator):
    model.eval()
    val_loss = MeanMetric().to(model.device)

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(val_dataloader),
        ):
                
            loss = model(batch["input_ids"])

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

            val_loss.update(loss_values["loss"])

    return val_loss


def train(args, accelerator):
    set_seed(0)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file("./scripts/UltraChat/template.txt")

    with accelerator.main_process_first():
        processor = UltraChatProcessor()
        dataset = processor.get_examples(args.data_file)

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=0)

    # wrapped_example = mytemplate.wrap_one_example(dataset[1])
    # print(wrapped_example)

    train_dataloader = PromptDataLoader(dataset=train_dataset, template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1024, decoder_max_length=1024,
        batch_size=2,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="head").dataloader

    val_dataloader = PromptDataLoader(dataset=val_dataset, template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1024, decoder_max_length=1024,
        batch_size=5,shuffle=False, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="head").dataloader


    # load the pipeline model PromptForGeneration.
    prompt_model = PromptForGeneration(plm=plm, template=mytemplate, tokenizer=tokenizer)

    device = accelerator.device
    prompt_model.to(device)


    optimizer = AdamW([p for p in prompt_model.parameters()if p.requires_grad], lr=args.lr, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader)*args.epochs)

    prompt_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(prompt_model, optimizer, train_dataloader, val_dataloader, scheduler)

    accelerator.register_for_checkpointing(scheduler)

    train_loss = MeanMetric().to(prompt_model.device)

    # training and generation.
    global_step = 0
    for epoch in range(args.epochs):
        for step, inputs in tqdm(enumerate(train_dataloader)):
            prompt_model.train()
            loss = prompt_model(inputs["input_ids"])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})
            train_loss.update(loss_values["loss"])
            global_step +=1

            
            if global_step %50 ==0:
                accelerator.save_state(f"ultrachat_{args.model}/step_{global_step}")

                val_loss = evaluate(args, prompt_model, val_dataloader, accelerator)

                log_train = {
                        "train_loss": train_loss.compute()
                    }
                log_val = {
                    "val_loss": val_loss.compute()
                }

                accelerator.print(f"Current LR: {scheduler.get_last_lr()[0]}")
                accelerator.print(format_metrics(log_train, "train", f" step {global_step} "))
                accelerator.print(format_metrics(log_val, "val", f" step {global_step} "))

                train_loss.reset()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.get_state_dict(prompt_model), f"ultrachat_{args.model}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model", type=str, default='gptj')
    parser.add_argument("--model_name_or_path", default='EleutherAI/gpt-j-6b')
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--data_file", default="./datasets/ultrachat_release_230407.json", type=str)
    args = parser.parse_args()
    # print(args)

    accelerator = Accelerator()

    train(args, accelerator)