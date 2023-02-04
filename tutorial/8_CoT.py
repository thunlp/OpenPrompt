# %%
from openprompt.data_utils.conditional_generation_dataset import CSQAProcessor
from openprompt.prompts import MixedTemplate
from openprompt.pipeline_base import PromptForGeneration, PromptDataLoader
from openprompt.plms import load_plm

use_cuda = True

processor = CSQAProcessor()
train_examples = processor.get_train_examples("../datasets/Reasoning/csqa")
print(train_examples[0])
# %%
examples_with_label = open("../scripts/CoT/csqa.txt").read()

template_text = examples_with_label + """\n\nQ: {\"placeholder\":\"text_a\"}\nAnswer Choices:\n{\"meta\":\"choices\", \"post_processing\": 'lambda x:\"|\".join([\"(\"+i[\"label\"]+\")\" + \" \" + i[\"text\"] for i in x])'}\nA: {\"mask\"}"""
print(template_text)
# %%
plm, tokenizer, model_config, wrapper = load_plm("gpt2", "gpt2-large")
# %%
template = MixedTemplate(plm, tokenizer, text=template_text)
wrapped_example = template.wrap_one_example(train_examples[0])
print(wrapped_example)

# %%
generation_arguments = {
    "max_new_tokens": 64,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
}

dataloader = PromptDataLoader(train_examples[:10], template, tokenizer=tokenizer, tokenizer_wrapper_class=wrapper, max_seq_length=1024, decoder_max_length=128, batch_size=1)
# %%
model = PromptForGeneration(plm, template, freeze_plm=True, plm_eval_mode=True, tokenizer=tokenizer)
if use_cuda:
    model = model.cuda()
# %%
for i, inputs in enumerate(dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    # print(inputs)
    _, output_sentence = model.generate(inputs, **generation_arguments)
    print("Context:", train_examples[i].text_a, "\n", train_examples[i].meta["choices"])
    print("Generated output:", output_sentence)
    print("True label:", train_examples[i].tgt_text)
    print()
    if i == 5:
        break

# %%
