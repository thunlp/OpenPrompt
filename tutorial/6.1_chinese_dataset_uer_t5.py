# In this scripts, we will teach you how to define a user defined
# TokenizerWrapper to do tasks that are not in the default configuration
# of

from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os
import re
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate


parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--project_root", default="/home/hushengding/OpenPrompt_CameraReady/OpenPrompt/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--result_file", type=str, default="../results.txt")
parser.add_argument("--max_steps", default=5000, type=int)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--eval_every_steps", type=int, default=500)
args = parser.parse_args()

args.result_file = os.path.join(args.project_root, args.result_file)

# Different from the other scripts, here the combination of tokenizer and model
# is not in the default configurations of openprompt.
# So we load the tokenizer and model separately.
from transformers import BertTokenizer, MT5ForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/t5-v1_1-base-chinese-cluecorpussmall")
model = MT5ForConditionalGeneration.from_pretrained("uer/t5-v1_1-base-chinese-cluecorpussmall")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
generated_text = text2text_generator("中国的首都是extra0京", max_length=50, do_sample=False)
print(generated_text)

# Then we slightly modify the Tokenizer wrapper to change the mask token to the BertTokenizer's mask token.
from openprompt.plms.seq2seq import T5TokenizerWrapper
class T5BertTokenizerWrapper(T5TokenizerWrapper):
    def mask_token(self, id):
        return f"extra{id}"

    def mask_token_ids(self, id):
        return self.tokenizer.convert_tokens_to_ids(f"extra{id}")
tokenizer.eos_token = "extra1"
tokenizer_wrapper = T5BertTokenizerWrapper(max_seq_length=128, tokenizer=tokenizer, decoder_max_length=5, decode_from_pad=False)



## Load a Chinese Classical Poetry Retrieval Dataset-Multiple Choice
## (http://cuge.baai.ac.cn/#/dataset?id=1&name=CCPM)
## In this task we choose the poetry that is best descripted by the translation.
## Please register in the web and download the dataset then put it into the <path> directory.
## A correct download have the follow file in "../CCPM"
## CCPM
## -- eval.py
## -- README-EN.md
## -- README-ZH.md
## -- test_public.jsonl
## -- train.jsonl
## -- valid.jsonl
##
# The final performance of this tasks using this scripts ~ 0.83, which can be better
# if we use better model such as CPM-2 etc.

def load_local_dataset(path="../CCPM/", split="train"):
    import json
    with open(f"{path}/{split}.jsonl", "r") as fin:
        L = fin.readlines()
    data =[]
    for json_str in L:
        result = json.loads(json_str)
        # from IPython import embed; embed()
        for idx, choice in enumerate(result['choices']):
            result[f'choice{idx}'] = choice
        result.pop("choices")
        data.append(InputExample(meta=result, label=int(result['answer'])))
    return data

dataset = {}
dataset['train'] = load_local_dataset(split="train")
dataset['val'] = load_local_dataset(split="valid")



from openprompt.prompts import ManualTemplate
mytemplate = ManualTemplate(tokenizer=tokenizer, text="""一：{"meta":"choice0"}； 二： {"meta":"choice1"}； 三： {"meta":"choice2"}； 四：{"meta":"choice3"}。哪一句表达了下面的意思: {"meta":"translation", "shortenable":True, "post_processing": lambda x:x.strip('。')}？第{"mask"}句。 """)
myverbalizer = GenerationVerbalizer(tokenizer, label_words={0:"一", 1:"二", 2: "三", 3:"四"}, is_rule=False)
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, verbalizer=myverbalizer, # be sure to add verbalizer
    tokenizer_wrapper=tokenizer_wrapper,
    batch_size=16,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="tail")
validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer,
    tokenizer_wrapper=tokenizer_wrapper,
    batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok
    truncate_method="tail")
# visualize some encoded data
print(tokenizer.decode(next(iter(train_dataloader))['input_ids'][0]))
print(tokenizer.decode(next(iter(train_dataloader))['decoder_input_ids'][0]))


## Below is the same precedure as the other scripts.
use_cuda = True
prompt_model = PromptForGeneration(plm=model,template=mytemplate, freeze_plm=False, plm_eval_mode=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()


print("truncate rate: {}".format(train_dataloader.tokenizer_wrapper.truncate_rate), flush=True)


generation_arguments = {
    "max_length": 2,
}

def evaluate(prompt_model, dataloader):
    predictions = []
    ground_truths = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
        predictions.extend(output_sentence)
        ground_truths.extend(inputs['tgt_text'])
    assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
    predictions = [prediction.strip() for prediction in predictions]
    ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    # shown one example
    print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
    score = sum([prediction[:len(ground_truth)]==ground_truth for prediction, ground_truth in zip(predictions, ground_truths)])/len(ground_truths)
    return score


from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor  # use Adafactor is the default setting for T5
loss_func = torch.nn.CrossEntropyLoss()

tot_step = args.max_steps


 # normally we freeze the model when using soft_template. However, we keep the option to tune plm
no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr)
scheduler1 = get_linear_schedule_with_warmup(
    optimizer1,
    num_warmup_steps=500, num_training_steps=tot_step)


tot_loss = 0
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False
gradient_accumulation_steps = 1

acc_traces = []
tot_train_time = 0
pbar_update_freq = 50
prompt_model.train()

pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(1000000):
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        tot_train_time -= time.time()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1

        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(pbar_update_freq)
                pbar.set_postfix({'loss': aveloss, "epoch": epoch})
                log_loss = tot_loss


                if optimizer1 is not None:
                    optimizer1.step()
                    optimizer1.zero_grad()
                if scheduler1 is not None:
                    scheduler1.step()

        tot_train_time += time.time()

        if actual_step % gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
            val_acc = evaluate(prompt_model, validation_dataloader)
            if val_acc >= best_val_acc:
                # torch.save(prompt_model.state_dict(),f"{args.project_root}/../ckpts/{this_run_unicode}.ckpt")
                best_val_acc = val_acc

            acc_traces.append(val_acc)
            print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
            prompt_model.train()

        if glb_step > args.max_steps:
            leave_training = True
            break

    if leave_training:
        break

