Base Classes
================================


Overview
------------------------------------
This page introduces the base classes of the prompt-learning framework. 
Generally, to conduct prompt-learning, a :obj:`PretrainedModel` is selected with the corresponding pre-trained task,
a :obj:`Template` class is established to wrap the original text,
and a :obj:`Verbalizer` class (if needed) is defined to project the labels to the label words in the vocabulary.
In OpenPrompt, the specific prompt-related classes will inherit these base classes. 


.. contents:: Contents
    :local:



Prompt Base
------------------------------------
Base classes of :obj:`Template` and :obj:`Verbalizer`.


.. autoclass:: openprompt.prompt_base.Template
   :members:

.. autoclass:: openprompt.prompt_base.Verbalizer
   :members:






Pipeline Base
----------------------------------

Base classes of :obj:`PromptDataLoader` and :obj:`PromptModel`, :obj:`PromptForClassification` and :obj:`PromptForGeneration`.


.. autoclass:: openprompt.pipeline_base.PromptDataLoader
   :members:

.. autoclass:: openprompt.pipeline_base.PromptModel
   :members:

.. autoclass:: openprompt.pipeline_base.PromptForClassification
   :members:

.. autoclass:: openprompt.pipeline_base.PromptForGeneration
   :members:

