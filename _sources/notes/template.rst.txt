.. _tutorial_template:

How to Write a Template?
=============================


As we stated, template (which could be specific textual tokens or abstract new tokens, the only difference is the initialization) 
is one of the most important module in a prompt-leanring framework.  In this tutorial, we introduce how to write a template and set the corresponding attributes for a ``Template`` class.

First, let's take a look what template could look like.



Examples of Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a template: 

.. _template_1:
.. code-block:: python

    "<text_a> It was a <mask> ."


This is also a template:

.. _template_2:
.. code-block:: python

    "<text_a> <new> <new> <new> <new> <mask> <new> <new> ."


This is also a template:

.. _template_3:
.. code-block:: python

    "<text_a> In this sentence, <meta:entity> is a <mask> ."

This is also a template:

.. _template_4:
.. code-block:: python

    "<text_a> the <mask> <meta:head> <mask> <mask> <mask> the <mask> <meta:tail> ."

This is also a template:


.. _template_5:
.. code-block:: python

   "<text_a> Question: <text_b> ? <soft>the Answer <soft>is <mask> ."


Kind of confusing? Don't worry, we will handle it step by step. 


Basic Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A template should firstly consider the original input, which is defaultly denoted as ``text_a``.
For tasks with multiple input text (or you can segment the text into multiple pieces), use other placeholders such as ``text_b``.

When defining a template, a series basic attributes may need to be considered. 
A template should contain at least one masked token that need to be predicted by PLMs, typically denoted as ``<mask>``. 
The ``loss_ids`` denotes which tokens' loss are computed.

.. code-block:: python
    
    template: ['<text_a>', 'it', 'is', '<mask>', '.']
    loss_ids: [0         , 0   , 0   , 1       , 0  ]


New tokens and soft tokens are tokens that will be optimized separately from the PLM, where new tokens are randomly initialized, and soft tokens are initialized by specific tokens in the vocabulary.
New tokens are denoted by ``new_ids`` and soft tokens are denoted by ``soft_ids``.

.. code-block:: python
    
    template: ['<text_a>', '<new>', '<new>', 'is', '<mask>', '<soft>a', '.']
    new_ids : [0         , 1      , 1      , 0   , 0       , 0        , 0  ]
    soft_ids: [0         , 0      , 0      , 0   , 0       , 1        , 0  ]


Meta Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the :py:class:`~openprompt.data_utils.data_utils.InputExample` class, we leave a ``meta`` API to store arbitrary extra information for the input example.

For example, for a relation extraction task, you want to add different head and tail entities in different input examples.

.. code-block:: python
    
    template = ["<text_a>", "In this sentence, the relation of", "<meta:head>", "and", "<meta:tail>", "is", "<mask>", "."]

In this case, with an :py:class:`~openprompt.data_utils.data_utils.InputExample`

.. code-block:: python
    
    {
        guid = 0,
        text_a = "Albert Einstein was born in Germany."
        meta = {
            "head": "Albert Einstein",
            "tail": "Germany"
        }

    }

The complete prompted example becomes

.. code-block:: python
    
    x = "Albert Einstein was born in Germany. In this sentence, the relation of Albert Einstein and Germany is <mask> ."


