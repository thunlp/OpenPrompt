import sys
sys.path.append(".")
sys.path.append("..")

from openprompt.data_utils import InputExample
dataset = [
    InputExample(
        text_a = "爱因斯坦非常的聪明",
        meta = {"entity": "爱因斯坦"},
        label = 0
    ),
    InputExample(
        text_a = "马云是阿里巴巴创始人",
        meta = {"entity": "马云"},
        label = 1
    ),
]

import bminf.torch as bt
from openprompt.plms.lm import LMTokenizerWrapper
plm = bt.models.CPM1()
tokenizer = plm.tokenizer
WrapperClass = LMTokenizerWrapper

from openprompt.prompts import MixedTemplate

mytemplate = MixedTemplate(
    model = plm,
    tokenizer = tokenizer,
    text = '{"placeholder": "text_a"} 上文中，{"meta": "entity"} 是一个{"mask"}',
)

wrapped_example = mytemplate.wrap_one_example(dataset[0]) 
print("Wrapped Example:", wrapped_example)

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=["学者", "商人"], prefix = '')
print("Verbalizer token id:", myverbalizer.label_words_ids.data)

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# ## below is standard training

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
next(iter(train_dataloader))

from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

for epoch in range(1):
    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        print("step: ", step)
        print("inputs: ", inputs)
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        print(tot_loss/(step+1))
    
# ## evaluate

# %%
from openprompt import PromptDataLoader

validation_dataloader = PromptDataLoader(dataset=dataset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=2, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

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

print(allpreds)
print(alllabels)
acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print("accuracy:", acc)
