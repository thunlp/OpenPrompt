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
from openprompt.plms.seq2seq import T5TokenizerWrapper
plm = bt.models.CPM2()
tokenizer = plm.tokenizer
WrapperClass = T5TokenizerWrapper

from openprompt.prompts import MixedTemplate

mytemplate = MixedTemplate(
    model = plm,
    tokenizer = tokenizer,
    text = '{"placeholder": "text_a"} 上文中，{"meta": "entity"} 是一个{"mask"}。', # TODO in chinese prefix space is of no use
)

wrapped_example = mytemplate.wrap_one_example(dataset[0]) 
print("Wrapped Example:", wrapped_example)

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=3, label_words=["<!>学者", "<!>商人"]) # TODO in chinese no prefix space

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# ## evaluate

# %%
from openprompt import PromptDataLoader

validation_dataloader = PromptDataLoader(dataset=dataset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, decode_from_pad=False,
    batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")


allpreds = []
alllabels = []
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
