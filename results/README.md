# Performance of Different Prompt Tuning Methods

We report the performance on widely-used datasets of each method.
Note that we do not attempt to match the exact performance score of
the referenced papers, if they use additional tricks such as data-augmentation
or prompt-ensemble. 

## Filed:
### Prompt
The config of the template.
### LM
The pre-trained language model we used.
### Ref
The specific yaml file or tutorial scripts to
achieve the results.
### Comment
Other noticable aspect of the experiment.



## webnlg_2017
The evaluation scripts: https://github.com/Yale-LILY/dart

| Prompt |      LM      |      Ref     | Comment | BLEU-SEEN | BLEU-UNSEEN | BLEU-ALL |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
|   Prf  | t5-base, fix, | [tutorial2.4](../tutorial/2.2_conditional_generation.py) |    plm-dropout-off      |   62.88   |    47.05    |   55.79  |
|   Prf  | t5-base, fix | [tutorial2.4](../tutorial/2.2_conditional_generation.py) |   plm-dropout-on      |   61.94   |    52.02    |   57.41  |
|    Prf |  gpt2-medium, fix,  |  [tutorial2.4](../tutorial/2.2_conditional_generation.py)    |  plm-dropout-off |    62.97       |    43.43     |      54.21     |
|    Prf |  gpt2-medium, fix |   [tutorial2.4](../tutorial/2.2_conditional_generation.py)   |  plm-dropout-on  |    60.21       |    45.67     |     53.66     |
