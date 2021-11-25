Prompt Generator
==================================

Overview
-------------------------------
This part contains `TemplateGenerator` and `VerbalizerGenerator`. Both follow the implementation in `Making Pre-trained Language Models Better Few-shot Learners(Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_ and conduct automatic generation of hard template and verbalizer based on the given counterpart.

Base Classes
---------------------------------
All prompt generator using LM-BFF method can be realized using the two base classes by simply re-implementing the abstract method in the two classes. The provided implementation ``T5TemplateGenerator`` and ``RobertaVerbalizerGenerator`` both inherit from the base classes, respectively.

.. autoclass:: openprompt.prompts.TemplateGenerator
   :members:

.. autoclass:: openprompt.prompts.VerbalizerGenerator
   :members:


T5TemplateGenerator
---------------------------------

.. autoclass:: openprompt.prompts.T5TemplateGenerator
   :members:


RobertaVerbalizerGenerator
---------------------------------

.. autoclass:: openprompt.prompts.RobertaVerbalizerGenerator
   :members: