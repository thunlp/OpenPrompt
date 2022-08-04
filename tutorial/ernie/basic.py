import pandas as pd
import paddle
import paddlenlp
paddle.set_device('cpu')
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from paddlenlp.transformers.ernie.modeling import ErnieForMaskedLM
from TokenizerWrapper import ErnieMLMTokenizerWrapper
from data_utils import InputExample, InputFeatures
from template import ErnieManualTemplate
from dataloader import ErniePromptDataLoader
from verbalizer import ErnieManualVerbalizer
from modeling import ErniePromptforClassification
from paddle.optimizer import AdamW

# load model
plm = ErnieForMaskedLM.from_pretrained('ernie-1.0')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
WrapperClass = ErnieMLMTokenizerWrapper

# load data
train = pd.read_csv('train.tsv',sep='\t')
dataset = {}
dataset['train'] = []
for index in train.index:
    input_example = InputExample(text_a = train.loc[index,'text_a'], text_b = '',label=int(train.loc[index,'label']))
    dataset['train'].append(input_example)


# make template
template_text = '{"placeholder": "text_a"}，感觉很{"mask"}.'
mytemplate = ErnieManualTemplate(tokenizer=tokenizer, text=template_text)


train_dataloader = ErniePromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=8,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

# make verbalizer
myverbalizer = ErnieManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[['差','坏','糟'],['棒','好']],post_log_softmax=True)



# init model
use_cuda = False
prompt_model = ErniePromptforClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()



loss_func = paddle.nn.loss.CrossEntropyLoss()
no_decay = ['bias', 'layer_norm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(learning_rate=1e-5,parameters=optimizer_grouped_parameters)

epochs = 3
for epoch in range(epochs):
    prompt_model.eval()
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader.dataloader):
        data = InputFeatures(input_ids=inputs[0],attention_mask=inputs[1],label=inputs[2],loss_ids=inputs[3])
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(data)
        labels = data['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.clear_grad()
        if step %1 ==0:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
