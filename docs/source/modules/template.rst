Templates
==================================

Overview
-------------------------------
The template is one of the most important module in prompt-learning, which wraps the original input with textual or soft-encoding sequence.

We implement common template classes in OpenPrompt. 


.. contents:: Contents
    :local:

.. currentmodule:: openprompt.prompts


Manual Template
---------------------------------
The basic manually defined textual template. 

.. autoclass:: openprompt.prompts.manual_template.ManualTemplate
   :members:


Prefix Template
---------------------------------
The template of prefix-tuning from `Prefix-Tuning: Optimizing Continuous Prompts for Generation <https://arxiv.org/abs/2101.00190>`_. 

.. autoclass:: openprompt.prompts.prefix_tuning_template.PrefixTuningTemplate
   :members:

Ptuning Template
---------------------------------
The template of P-tuning from `GPT understands, too. <https://arxiv.org/pdf/2103.10385.pdf>`_. 

.. autoclass:: openprompt.prompts.ptuning_prompts.PtuningTemplate
   :members:


PTR Template
--------------------
The template of PTR from `PTR: Prompt Tuning with Rules for Text Classification <https://arxiv.org/abs/2105.11259>`_.

.. autoclass:: openprompt.prompts.ptr_prompts.PTRTemplate
   :members:

Mixed Template
--------------------
Our newly introduced mixed template class to flexibly define your templates.

.. autoclass:: openprompt.prompts.mixed_template.MixedTemplate
   :members: