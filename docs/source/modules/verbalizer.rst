Verbalizer
==================================

Overview
-------------------------------
The verbalizer is one of the most important module in prompt-learning, which projects the original labels to a set of label words.

We implement common verbalizer classes in OpenPrompt. 


.. contents:: Contents
    :local:

.. currentmodule:: openprompt.prompts

One to One Verbalizer
---------------------------------
The basic one to one Verbalizer. 

.. autoclass:: openprompt.prompts.one2one_verbalizer.One2oneVerbalizer
   :members:

Manual Verbalizer
---------------------------------
The basic manually defined Verbalizer. 

.. autoclass:: openprompt.prompts.manual_verbalizer.ManualVerbalizer
   :members:


Automatic Verbalizer
---------------------------------
The Automatic Verbalizer defined in `Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification <https://arxiv.org/pdf/2010.13641.pdf>`_. 

.. autoclass:: openprompt.prompts.automatic_verbalizer.AutomaticVerbalizer
   :members:

Knowledgeable Verbalizer
---------------------------------
The Knowledgeable Verbalizer defined in `Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification <https://arxiv.org/pdf/2108.02035.pdf>`_. 

.. autoclass:: openprompt.prompts.knowledgeable_verbalizer.KnowledgeableVerbalizer
   :members:

PTR Verbalizer
---------------------------------
The verbalizer of PTR from `PTR: Prompt Tuning with Rules for Text Classification <https://arxiv.org/abs/2105.11259>`_.

.. autoclass:: openprompt.prompts.ptr_prompts.PTRVerbalizer
   :members:

Generation Verbalizer
---------------------------------
This verbalizer empower the "generation for all the tasks" paradigm. 

.. autoclass:: openprompt.prompts.generation_verbalizer.GenerationVerbalizer
   :members:


Soft Verbalizer
---------------------------------

.. autoclass:: openprompt.prompts.soft_verbalizer.SoftVerbalizer
   :members: