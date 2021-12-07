import sys
sys.path.append(".")
sys.path.append("..")

from openprompt.data_utils import InputExample
from openprompt.data_utils.ZH import CMeEE_NER
processor = CMeEE_NER()
trainset = processor.get_train_examples("datasets/ZH/CMeEE_NER")
devset = processor.get_dev_examples("datasets/ZH/CMeEE_NER")

import bminf.torch as bt
from openprompt.plms.lm import CPM2TokenizerWrapper
plm = bt.models.CPM2()
tokenizer = plm.tokenizer
WrapperClass = CPM2TokenizerWrapper

from openprompt.prompts import MixedTemplate

mytemplate = MixedTemplate(
    model = plm,
    tokenizer = tokenizer,
    text = '{"meta": "context", "shortenable": True} 上文中，{"meta": "entity"} 是一个{"mask"}。选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])}',
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

# from openprompt import PromptDataLoader

# train_dataloader = PromptDataLoader(dataset=trainset, template=mytemplate, tokenizer=tokenizer, 
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=10, 
#     batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="head")
# # next(iter(train_dataloader))

# from transformers import  AdamW, get_linear_schedule_with_warmup
# loss_func = torch.nn.CrossEntropyLoss()

# no_decay = ['bias', 'LayerNorm.weight']

# # it's always good practice to set no decay to biase and LayerNorm parameters
# optimizer_grouped_parameters1 = [
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]

# # Using different optimizer for prompt parameters and model parameters
# optimizer_grouped_parameters2 = [
#     {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
# ]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
# optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

# for epoch in range(1):
#     tot_loss = 0 
#     for step, inputs in enumerate(train_dataloader):
#         if use_cuda:
#             inputs = inputs.cuda()
#         logits = prompt_model(inputs)
#         labels = inputs['label']
#         loss = loss_func(logits, labels)
#         loss.backward()
#         tot_loss += loss.item()
#         optimizer1.step()
#         optimizer1.zero_grad()
#         optimizer2.step()
#         optimizer2.zero_grad()
#         print(loss.item(), tot_loss/(step+1))
    
# ## evaluate

# %%
from openprompt import PromptDataLoader

validation_dataloader = PromptDataLoader(dataset=devset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=10,
    batch_size=2, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

prompt_model = prompt_model.eval()

allpreds = []
alllabels = []
with torch.no_grad():
    for step, inputs in enumerate(validation_dataloader):
        print("step: ", step)
        print("input shape: ", inputs['input_ids'].shape)
        print("inputs: ", inputs)
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
