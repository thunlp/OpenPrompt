Introduction with an Example
================================================




How to construct a prompt-learning pipeline? 
With the modularity and flexibility of OpenPrompt, you can easily develop a prompt-learning application step by step.


Step 1. Define a task
----------------------------------------------------
The first step is to determine the current NLP task, think about what's your data looks like and what do you want from the data!
That is, the essence of this step is to determine the ``classses`` and the :py:class:`~openprompt.data_utils.data_utils.InputExample` of the task.
For simplicity, we use Sentiment Analysis as an example. 

You can also use our pre-defined :ref:`data_processors` to get train/dev/test dataset for a given task.

..  code-block:: python

    from openprompt.data_utils import InputExample
    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "negative",
        "positive"
    ]
    dataset = [ # For simplicity, there's only two examples
        # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
        InputExample(
            guid = 0,
            text_a = "Albert Einstein was one of the greatest intellects of his time.",
        ),
        InputExample(
            guid = 1,
            text_a = "The film was badly made.",
        ),
    ]



Step 2. Obtain a PLM
--------------------------------------------------------------------------------------------------------

Choose a PLM to support your task. Different models have different attributes, 
we encourge you to use OpenPrompt to explore the potential of various PLMs.
OpenPrompt is compatible with models on `huggingface <https://huggingface.co/transformers/>`_, 
the following models have been tested:

* Masked Language Models (MLM): ``BERT``, ``RoBERTa``, ``ALBERT``
* Autoregressive Language Models (LM): ``GPT``, ``GPT2``
* Sequence-to-Sequence Models (Seq2Seq): ``T5``

Simply use a ``get_model_class`` to obtain your PLM.

..  code-block:: python

    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

Step 3. Define a Template
--------------------------------------------------------------------------------------------------------

A ``Template`` is a modifier of the original input text, which is also one of the most important modules in prompt-learning.
A more advanced tutorial to define a template is in :ref:`tutorial_template`

Here is an example, where the ``<text_a>`` will be replaced by the :obj:`text_a` in :py:class:`~openprompt.data_utils.data_utils.InputExample`, and the ``<mask>`` will be used to predict a label word.

..  code-block:: python

    from openprompt.prompts import ManualTemplate
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} It was {"mask"}',
        tokenizer = tokenizer,
    )

Step 4. Define a Verbalizer
--------------------------------------------------------------------------------------------------------

A ``Verbalizer`` is another important (but not necessary such as in generation) in prompt-learning,which projects the original labels (we have defined them as ``classes``, remember?) to a set of label words.
A more advanced tutorial to define a verbalizer is in :ref:`How_to_write_a_verbalizer`


Here is an example that we 

* project the ``negative`` class to the word  `bad`
* project the ``positive`` class to the words `good`, `wonderful`, `great`.

..  code-block:: python

    from openprompt.prompts import ManualVerbalizer
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "negative": ["bad"],
            "positive": ["good", "wonderful", "great"],
        },
        tokenizer = tokenizer,
    )


Step 5. Construct a PromptModel
--------------------------------------------------------------------------------------------------------

Given the task, now we have a ``PLM``, a ``Template`` and a ``Verbalizer``, we combine them into a ``PromptModel``.

Note that although this example naively combine the three modules, you can actually define some complicated interactions among them.


..  code-block:: python

    from openprompt import PromptForClassification
    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )


Step 6. Define a DataLoader
--------------------------------------------------------------------------------------------------------

A ``PromptDataLoader`` is basically a prompt version of pytorch Dataloader, which also includes a ``Tokenizer`` and a ``Template``.




..  code-block:: python

    from openprompt import PromptDataLoader
    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer, 
        template = promptTemplate, 
        tokenizer_wrapper_class=WrapperClass,
    )


Step 7. Train and inference
--------------------------------------------------------------------------------------------------------

Done! We can conduct training and inference the same as other processes in Pytorch.


..  code-block:: python
    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            print(classes[preds])
    # predictions would be 1, 0 for classes 'positive', 'negative'

This is a quick start of OpenPrompt, please refer to the APIs for more details.
