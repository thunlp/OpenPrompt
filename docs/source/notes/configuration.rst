.. _ref_config:

Play with Configuration
==============================

OpenPrompt suggests to use configuration file to developen the users' own prompt-leanring pipelines.
We provide a ``config_default.yaml`` file to implement common attributes in prompt-learning, which will be detailed introduced next.
For a prompt-learning pipeline derived by OpenPrompt, you can implement a model specific config file to implement specific attributes.


We provide a unified entrance for experiment with OpenPrompt. Just run the following code  in the root directory.

.. code-block:: python

    python experiments/cli.py --config_yaml experiments/classification_manual_prompt.yaml

You may choose the configuration file we wrote for you, or write your own configuration file.


Next, we will introduce the meaning of each configuration parameter in the default configuration.
In your own experiments, you can create a yaml file containing a subset of configuration parameters
that you want to change or specify. 


Default Configuration
------------------------------
Now we introduce the details of the default configuration of OpenPrompt.


Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``environment``, we can set the attributes regarding to the training and inference environment.

- **num_gpus** (:obj:`int`): The number of GPUs during training and evaluation.
- **cuda_visible_devices**: Which devices are visible during training and evaluation.
- **local_rank:** The indices of devices for the current process.

Example:

.. code-block:: yaml

    environment:
        num_gpus: 1
        cuda_visible_devices:
            - 0
        local_rank: 0


Reproduce
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `reproduce` configuration controls key attributes that determines the reproduction of a prompt-leanring framework.
Specifically, seeds for all potential randomness.

Example:

- **seed**: If seed this seed, and other seeds are unset, then all the seeds will use this value.

.. code-block:: yaml

    reproduce:
        seed: 100


PLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PLM`` implements attributes regarding to pre-trained language models, including the model's type, path and optimization.

- **model_name**: The name of the pre-trained model.
- **model_path**: The path of the pre-trained model. 
- **optimize**: 
    - **freeze_para**:  If the parameters of the model are freezed.
    - **loss_function**:  The loss function during training.
    - **no_decay**: The ``no_decay`` setup of the optimization.
    - **lr**: The learning rate during training.
    - **weight_decay**: The ``weight_decay`` setup of optimization.
    - **scheduler**: 
        - **type**: The scheduler type. 
        - **num_warmup_steps**: The number of steps for warming up.

Example:

.. code-block:: yaml

    plm:
        model_name: 
        model_path:
        optimize: 
            freeze_para: False
            no_decay:
                - bias
                - LayerNorm.weight
            lr: 0.0005
            weight_decay: 0.01
            scheduler:
                type: 
                num_warmup_steps: 500

Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This part contains the attributes of ``train``, ``dev`` and ``test``. 

- **train**
    - **num_epochs**: The number of epochs during training.
    - **batch_size**: The batch size during training. 
    - **shuffle_data**: If True, the data will be shuffled during training.
    - **teacher_forcing**: If True, the teach forcing method will be used during training.
    - **clean**: If True, not saving checkpoints and not logging tensorboard. However, test will use the last model but not the best model in validation.
- **dev**
    - **batch_size**: The batch size during validation. 
    - **shuffle_data**: If True, the data will be shuffled during validation.
- **test**
    - **batch_size**: The batch size during testing. 
    - **shuffle_data**: If True, the data will be shuffled during testing.

Example:

.. code-block:: yaml

    train:
        num_epochs: 5
        batch_size: 2
        shuffle_data: False
        teacher_forcing: False
        clean: False

    dev:
        batch_size: 2
        shuffle_data: False

    test:
        batch_size: 2
        shuffle_data: False

Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The configuration about the current task.
There will be a parent configuration ``task`` to determine the current type of task, e.g. classfiication.
And for the specific task, a user could specifically set the corresponding attributes.


Example:

.. code-block:: yaml

    task: classification
    classification:
        parent_config: task
        metric: 
            - micro-f1
        loss_function: cross_entropy ## select from cross_entropy

    generation:
        parent_config: task
        gen_max_length: 128
        decoding_strategy: greedy
    
    relation_classification:
        parent_config: task

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Dataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the configuration about the dataloader,
which sets some attributes like ``max_seq_length``, etc.

Example:

.. code-block:: yaml

    dataloader:
        max_seq_length: 256
        decoder_max_length: 256
        predict_eos_token: False  # necessary to set to true in generation.
        truncate_method: "head" # choosing from balanced, head, tail

Learning Setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration about the learning settings, including ``full``, ``few-shot`` and ``zero-shot``.

.. code-block:: yaml

    learning_setting:   # selecting from "full", "zero-shot", "few-shot"

    zero_shot:
        parent_config: learning_setting

    few_shot:
        parent_config: learning_setting
        few_shot_sampling:
    
    sampling_from_train:
        parent_config: few_shot_sampling
        num_examples_per_label: 10
        also_sample_dev: True
        num_examples_per_label_dev: 10
        seed:
            - 123
            - 456


Prompt-specific Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration about templates and verbalizers, there are different attributes for different classes.
Here are some examples:

.. code-block:: yaml
   
    template:
    verbalizer:

    manual_template:
        parent_config: template
        text: 
        mask_token: <mask>
        placeholder_mapping:
        <text_a>: text_a
        <text_b>: text_b
    file_path:
        choice: 0
        optimize:  # the parameters related to optimize the template


    automatic_verbalizer:
        parent_config: verbalizer
        num_cadidates: 1000
        label_word_num_per_class: 1
        num_searches: 1
        score_fct: llr
        balance: true
    optimize:
        level: epoch
    num_classes:
        init_using_split: valid

    one2one_verbalizer:
        parent_config: verbalizer
        label_words:
        prefix: " "
        multi_token_handler: first
        file_path:
        choice:
        num_classes:
        optimize:
  
    manual_verbalizer:
        parent_config: verbalizer
        label_words:
        prefix: " "
        multi_token_handler: first
        file_path:
        choice:
        num_classes:
        optimize:

    prefix_tuning_template:
        parent_config: template
        text:
            mask_token: <mask>
        num_token: 5
        placeholder_mapping: 
            <text_a>: text_a
            <text_b>: text_b
        prefix_dropout: 0.0
        optimize:
            lr: 0.0001