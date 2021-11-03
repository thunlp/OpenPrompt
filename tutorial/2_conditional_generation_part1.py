# %% [markdown]
# Now it's time to step into generation tasks.
# %% [markdown]
# We provide 

# %%

import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor

# Don't forget to download the dataset first
dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("./datasets/CondGen/webnlg_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("./datasets/CondGen/webnlg_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("./datasets/CondGen/webnlg_2017/")

# print(train_dataset[0])


# %% [markdown]
# ## Construct Template
# 
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
# You can load the plm related things provided by openprompt simply by calling:

# %%
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")


# %% [markdown]
# # Try more prompt!

# You can use templates other than manual template, for example the soft template.
# In SoftTemplate, the text is prepended by `num_tokens` soft template tokens before the input samples. 

from openprompt.prompts import SoftTemplate

mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} Make a sentence {"mask"}',num_tokens=100)

# mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.')

# mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? Is it correct? {"mask"}.')

# To better understand how does the template wrap the example, we visualize one instance.


wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)


# Now, the wrapped example is ready to be pass into the tokenizer, hence producing the input for language models.
# You can use the tokenizer to tokenize the input by yourself, but we recommend using our wrapped tokenizer, which is a wrapped tokenizer tailed for InputExample. 
# The wrapper has been given if you use our `load_plm` function, otherwise, you should choose the suitable wrapper based on
# the configuration in `openprompt.plms.__init__.py`.

wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=128, tokenizer=tokenizer,truncate_method="head")
# or
from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=128, tokenizer=tokenizer,truncate_method="head")


tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=True) # when setting teacher_forcing=True, the mask will be filled with tgt_text
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

model_inputs = {}
for split in ['train', 'validation', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        if split == 'train':
            teacher_forcing=True
        else:
            teacher_forcing=False
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=teacher_forcing)
        model_inputs[split].append(tokenized_example)


from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=4,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")
# next(iter(train_dataloader))

from openprompt import PromptForGeneration

use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW


# Only tune the soft template.
optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)


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
    print("validation score", score, flush=True)
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


# training and generation.
for epoch in range(10):
    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        
    evaluate(prompt_model, validation_dataloader)

generated_sentence = evaluate(prompt_model, test_dataloader)  # sentence_score ~ 52

with open("../../Generated_sentence_webnlg.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")


