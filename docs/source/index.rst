:github_url: https://github.com/ningding97/OpenPrompt/tree/master/openprompt

OpenPrompt Documentation
=============================
Prompt-learning is the latest paradigm to adapt pre-trained language models (PLMs) to downstream NLP tasks, which modifies the input text with a textual template and directly uses PLMs to conduct pre-trained tasks. 
*OpenPrompt* is a library built upon `PyTorch <https://pytorch.org/>`_ and provides a standard, flexible and extensible framework to deploy the prompt-learning pipeline.
OpenPrompt supports loading PLMs directly from `huggingface transformers <https://huggingface.co/transformers/>`_. 
In the future, we will also support PLMs implemented by other libraries.

If you find OpenPrompt useful, please kindly cite our `paper <https://arxiv.org/abs/2111.01998>`_.

.. code-block:: bibtex

   @article{ding2021openprompt,
      title={OpenPrompt: An Open-source Framework for Prompt-learning},
      author={Ding, Ning and Hu, Shengding and Zhao, Weilin and Chen, Yulin and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong},
      journal={arXiv preprint arXiv:2111.01998},
      year={2021}
   }


.. image:: demo.gif

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   notes/installation
   notes/examples
   notes/prompt
   notes/template
   notes/verbalizer
   notes/faq
   



.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/base
   modules/template
   modules/verbalizer
   modules/prompt_generator
   modules/data_utils
   modules/data_processors
   modules/trainer
   modules/utils
   notes/configuration



Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`