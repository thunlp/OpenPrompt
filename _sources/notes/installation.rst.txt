Installation
========================

OpenPrompt is tested on `Python 3.8 <https://www.python.org/>`_ and `Pytorch 1.9 <https://pytorch.org/>`_. Currently, the pre-trained language models and tokenizers are loaded from `huggingface transformers <https://huggingface.co/transformers/>`_. 
And OpenPrompt will support models and tokenizers implemented by other libraries in the future. 


.. note::
    Please do not conduct the installation as a root user on your system Python.
    We recommend setup a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_ environment or create a `Docker image <https://www.docker.com/>`_.


Installation from GitHub
----------------------------
From our github repository, you can install the latest version of OpenPrompt.

.. code-block:: bash

    git clone https://github.com/thunlp/OpenPrompt.git
    cd OpenPrompt
    python setup.py install

Now the OpenPrompt package is installed, let's get started.


