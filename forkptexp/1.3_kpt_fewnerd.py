# In this scripts, you will laern how to do calibartion and zero-shot learning
# We use manual verbalizer and knowledgeable verbalizer as examples.
from tqdm import tqdm
from openprompt.data_utils.typing_dataset import FewNERDProcessor
import torch
from openprompt.data_utils.utils import InputExample

dataset = {}
dataset['train'] = FewNERDProcessor().get_train_examples("./datasets/Typing/FewNERD")
dataset['dev'] = FewNERDProcessor().get_dev_examples("./datasets/Typing/FewNERD")
dataset['test'] = FewNERDProcessor().get_test_examples("./datasets/Typing/FewNERD")
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")

from openprompt.prompts import ManualTemplate
mytemplate = ManualTemplate(tokenizer=tokenizer).from_file("scripts/Typing/FewNERD/manual_template.txt", choice=0)

from openprompt import PromptDataLoader



# ## Define the verbalizer
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
class_labels = FewNERDProcessor().get_labels()
myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=0.3).from_file("scripts/Typing/FewNERD/knowledgeable_verbalizer.json")

# from IPython import embed
# embed()
# or
# myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file("scripts/Typing/FewNERD/fewnerd_verbalizer.json")


# from IPython import embed
# embed()
# (contextual) calibration
from openprompt.data_utils.data_sampler import FewShotSampler
support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
dataset['support'] = support_sampler(dataset['train'], seed=1)

# or template-only calibration
# make an pseudo-example:
# dataset['support'] = [InputExample(text_a="",text_b="")] # uncomment this line to use template-only calibration

for example in dataset['support']:
    example.label = -1 # remove the labels of support set for clarification
support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
from openprompt.utils.calibrate import calibrate
# calculate the calibration logits
cc_logits = calibrate(prompt_model, support_dataloader)
print("the calibration logits is", cc_logits)

# register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
# currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.
prompt_model.verbalizer.register_calibrate_logits(cc_logits)
new_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
print("Original number of label words per class: {} \n After filtering, number of label words per class: {}".format(org_label_words_num, new_label_words_num))

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
allpreds = []
alllabels = []
pbar = tqdm(test_dataloader)
for step, inputs in enumerate(pbar):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print("test:", acc)  # roughly ~0.853 when using template 0