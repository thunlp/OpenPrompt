import sys
sys.path.append(".")
sys.path.append("..")

from openprompt.data_utils import InputExample
from openprompt.data_utils.ZH import ChnSentiCorp
from openprompt.data_utils.data_sampler import FewShotSampler
processor = ChnSentiCorp()
# TODO other chinese datasets are not fully adapted yet
trainset = processor.get_train_examples("datasets/ZH/ChnSentiCorp")
devset = processor.get_dev_examples("datasets/ZH/ChnSentiCorp")
# sampler  = FewShotSampler(num_examples_per_label=8, num_examples_per_label_dev=8, also_sample_dev=True)
# trainset, devset = sampler(trainset, devset)

import bminf.torch as bt
use_cpm_version = 2
if use_cpm_version == 1:
    from openprompt.plms.lm import LMTokenizerWrapper
    plm = bt.models.CPM1()
    tokenizer = plm.tokenizer
    WrapperClass = LMTokenizerWrapper
elif use_cpm_version == 2:
    from openprompt.plms.seq2seq import CPM2TokenizerWrapper
    plm = bt.models.CPM2()
    tokenizer = plm.tokenizer
    WrapperClass = CPM2TokenizerWrapper

from openprompt.prompts import SoftTemplate, MixedTemplate

mytemplate = SoftTemplate(
    model = plm,
    tokenizer = tokenizer,
    # text = '{"meta": "context", "shortenable": True} 上文中，{"meta": "entity"} 是一个{"mask"}。选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])}',
    # text = '前提：{"meta": "premise", "shortenable": True} 假设: {"meta": "hypothesis", "shortenable": True} 问题：前提和假设是什么关系? 选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])} 回答:{"mask"}',
    text = '文本：{"meta": "context", "shortenable": True} 问题:上述文本所表达的情感是积极的还是消极的? 回答：{"mask"}',
)

wrapped_example = mytemplate.wrap_one_example(trainset[0])
print("Wrapped Example:", wrapped_example)

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
label_words = processor.labels_mapped
myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label_words), label_words=label_words, prefix = '')
print("Verbalizer token id:", myverbalizer.label_words_ids.data)

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# ## below is standard training

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=trainset, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=8,
    batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# next(iter(train_dataloader))

validation_dataloader = PromptDataLoader(dataset=devset, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=8,
    batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

print("names: ", [n for n, p in prompt_model.plm.named_parameters()])
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

print("names: ", [n for n, p in prompt_model.template.named_parameters()])
# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    # {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    {'params': [p for n,p in prompt_model.template.named_parameters()]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=0)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-1/1024)

for epoch in range(3):
    # ## train
    prompt_model.train()

    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)*1024
        loss.backward()
        # print(prompt_model.template.soft_embeds.grad)
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        print(f"epoch {epoch} - step {step}: ", loss.item(), tot_loss/(step+1))

    # ## evaluate

    prompt_model = prompt_model.eval()

    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(validation_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            print("step :", step)

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("accuracy:", acc)

