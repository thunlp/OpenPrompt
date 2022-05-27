.. _tutorial_template:

How to Write a Template?
=============================


As we stated, 
template (which could be specific textual tokens or abstract new tokens, 
the only difference is the initialization) 
is one of the most important module in a prompt-leanring framework.  
In this tutorial, we introduce how to write a template and set the corresponding attributes for a ``Template`` class.


Our template language takes the insight from the Dict grammar from Python in order to make it easy-to-learn. 
We use a ``meta`` key to denote the original text input, or the part of the input, or other key information.
A ``mask`` key is used to denote the indice of the token that need to be predicted. A ``soft`` key denotes soft tokens and textual tokens could be directly written down.

Textual Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


A simple template for binary sentiment classification, the ``sentence`` denotes the original input and the ``mask`` is the target position,

.. _template_0:
.. code-block:: python

    {"meta": "sentence"}. It is {"mask"}.


Here is a basic template for news topic classification, where one example contains two parts -- a ``title`` and a ``description``,

.. _template_1:
.. code-block:: python

    A {"mask"} news : {"meta": "title"} {"meta": "description"}

In entity typing, an ``entity`` is a key information, and we want to copy it in the template,s

.. _template_2:
.. code-block:: python

    {"meta": "sentence"} {"text": "In this sentence,"} {"meta": "entity"} {"text": "is a"} {"mask"},

    # you can also omit the `text` key
    {"meta": "sentence"}. In this sentence, {"meta": "entity"} is a {"mask"},
    


Easy, huh? We can also specify that in topic classification, the title should not be truncated,

.. _template_3
.. code-block:: python

    a {"mask"} news: {"meta": "title", "shortenable": False} {"meta": "description"}

Soft & Mix Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enough for the textual template, let's try some soft tokens, if you use ``{'soft'}``, 
the token will be randomly initialized. If you add some textual tokens at the value position,
the soft token(s) will be initialized by these tokens. 
Note that, a textual template will optimized with the model. 
And soft tokens will be separately optimzed. 

.. _template_4
.. code-block:: python

    {"meta": "premise"} {"meta": "hypothesis"} {"soft": "Does the first sentence entails the second?"} {"mask"} {"soft"}.

We can also mix them up, too, note that if two soft tokens have same ``soft_ids``, they will share embeddings,

.. _template_5
.. code-block:: python

    {"meta": "premise"} {"meta": "hypothesis"} {"soft": "Does"} {"soft": "the", "soft_id": 1} first sentence entails {"soft_id": 1} second?

If you try to define 10000 soft tokens, please use the key ``duplicate``,

.. _template_6
.. code-block:: python

    {"soft": None, "duplicate": 10000} {"meta": "text"} {"mask"}


If you try to define 10000 identical soft tokens, use the key `same`,

.. _template_7
.. code-block:: python

    {"soft": None, "duplicate": 10000, "same": True}

Post processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also support post-processing (e.g. write an lambda expression to strip the final punctuation in data),

.. _template_8
.. code-block:: python

    {"meta": 'context', "post_processing": lambda s: s.rstrip(string.punctuation)}. {"soft": "It was"} {"mask"}

You can also apply an MLP to post process your tokens,

.. _template_9
.. code-block:: python

    {"text": "This sentence is", "post_processing": "mlp"} {"soft": None, "post_processing": "mlp"} 


Our flexible template language support token-level specifying in prompt-learning, 
you can easily develop complex desired template by OpenPrompt, 
try it out!
