# Performance of Different Prompt Tuning Methods

We report the performance on widely-used datasets of each method.
Note that we do not attempt to match the exact performance score of
the referenced papers, if they use additional tricks such as data-augmentation
or prompt-ensemble. 

## Table Heads Explanation
### Prompt
The config of the template.
### LM
The pre-trained language model we used.
### Ref
The specific yaml file or tutorial scripts to
achieve the results.
### Comment
Other noticeable aspects of the experiments.


-------

## Few-NERD

Dataset details see https://arxiv.org/abs/2105.07464
N-S means N-shot

|Prompt| LM| Ref |Comment | Acc(8-S) | MiF(8-S)|
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|
|ManualT+ManualV| bert-base-cased|[yaml](../referenced_yamls/1107032409468965.yaml)| | 55.30| 67.88|




## webnlg_2017
The evaluation scripts: https://github.com/Yale-LILY/dart

| Prompt |      LM      |      Ref     | Comment | BLEU-SEEN | BLEU-UNSEEN | BLEU-ALL |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
|   Prf  | t5-base, fix, | [tutorial2.2](../tutorial/2.2_conditional_generation.py) |    plm-dropout-off      |   62.88   |    47.05    |   55.79  |
|   Prf  | t5-base, fix | [tutorial2.2](../tutorial/2.2_conditional_generation.py) |   plm-dropout-on      |   61.94   |    52.02    |   57.41  |
|    Prf |  gpt2-medium, fix,  |  [tutorial2.2](../tutorial/2.2_conditional_generation.py)    |  plm-dropout-off |    62.97       |    43.43     |      54.21     |
|    Prf |  gpt2-medium, fix |   [tutorial2.2](../tutorial/2.2_conditional_generation.py)   |  plm-dropout-on  |    60.21       |    45.67     |     53.66     |


## SuperGLUE


### All result
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py)\* | Generation Objective|  0.74 |

\*
A command line command to reproduce all results:
```bash
python tutorial/4.1_all_tasks_are_generation.py --model t5-lm --plm_eval_mode --dataset $datasetname --template_id 0 --verbalizer_id 0 --seed 100 --prompt_lr 0.3 --optimizer Adafactor --warmup_step_prompt 0 --max_steps 20000 --eval_every_steps 500
```

### Boolq
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | manual_0 | [tutorial](../tutorial/1.4_soft_template.py) | Classification Objective|  0.833|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.825|

### MultiRC
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | manual_0 | [tutorial](../tutorial/1.4_soft_template.py) | Classification Objective|  0.812|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.797 |

### WiC
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | manual_0 | [tutorial](../tutorial/1.4_soft_template.py) | Classification Objective|  0.701|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.650 |

### CB
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.75 |

### RTE
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | manual_0 | [tutorial](../tutorial/1.4_soft_template.py) | Classification Objective| 0.820 |
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.794 |

### WSC
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | gen_0\* | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.625 |

\* The verbalier `[{"text": "Another word}, {"meta": "span1_text"}]` Might not be the optimal, just to show a use case of the generation verbalizer.

### COPA
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | gen_0\*| [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.72 |

\* The verbalizer `[{"meta":"choice1"}, {"meta":"choice2"}]` is different from the verbalizer used in T5, `["True", "False"]`. Superisingly, recovering the whole choice1/choice2 sentence is very easy for LM, and yield much better result (0.72 vs 0.60)

### RECORD
| Prompt | LM  | Template | Verbalizer | Ref | Comment | Validation Acc |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
| Soft | t5-lg-lm-ad| manual_0 | gen_0 | [tutorial](../tutorial/4.1_all_tasks_are_generation.py) | Generation Objective|  0.770 |





