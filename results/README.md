# Performance of Different Prompt Tuning Methods

We report the performance on widely-used datasets of each method.
Note that we do not attempt to match the exact performance score of
the referenced papers, if they use additional tricks such as data-augmentation
or prompt-ensemble. The performance we are reporting is the property of the
specific template, verbalizer, combination, etc.

## agnews

| setting | referenced_paper | shot  | 0 | 1 | | 8 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| ManualTemplate |  PET| | | | |

## yahoo-questions-topics


## superglue.boolq


## webnlg_2017

| Prompt |      LM      |      Ref     | Comment | BLEU-SEEN | BLEU-UNSEEN | BLEU-ALL |
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|
|   Prf  | t5-base, fix | [tutorial2.4](../tutorial/2.4_conditional_generation_prefixtuning2.py) |         |   62.73   |    48.00    |   56.04  |
|    Prf |  gpt2-medium, fix ||         |    59.42       |    40.96     |      51.13      |