
# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation, 
# Which we do not include in it. 

import argparse

from torch.utils.data import dataloader
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
args = parser.parse_args()

from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("./datasets/CondGen/webnlg_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("./datasets/CondGen/webnlg_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("./datasets/CondGen/webnlg_2017/")


# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function 
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")


# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_pro import PrefixTuningTemplatePro
# we can use a plain text as the default setting
# i.e. 
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to 
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
mytemplate = PrefixTuningTemplatePro(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} </s> {"mask"} ', using_decoder_past_key_values=True)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your tempalte doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")


# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training. 
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]


optimizer = AdamW(optimizer_grouped_parameters2, lr=args.lr)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function 
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence


generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": None
}

import torch
# training and generation.
for epoch in range(5):
    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters2[0]['params'], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        
    evaluate(prompt_model, validation_dataloader)

generated_sentence = evaluate(prompt_model, test_dataloader)

with open("../../Generated_sentence_webnlg_t5.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")

