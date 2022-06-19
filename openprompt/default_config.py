from yacs.config import CfgNode

def get_default_config():
    # OpenPrompt's default configuration options
    cfg = CfgNode(new_allowed=True)

    # ENVIRONMENT
    ###################################
    cfg.environment = CfgNode(new_allowed=True)
    cfg.environment.num_gpus = 1 # number of gpus to use
    cfg.environment.cuda_visible_devices = [0] # which index of cuda devices is visible to the program
    cfg.environment.local_rank = 0 # the main device in the cuda visible devices that your DataParallel model will put the model on.
                    # The following should holds: local_rank < len(cuda_visible_devices)
    cfg.environment.model_parallel = False  # whether to perform model parallel
    cfg.environment.device_map = None  # the device_map, such as "{0: [0, 1, 2], 1: [3, 4, 5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14, 15, 16],3: [17, 18, 19, 20, 21, 22, 23]}

    cfg.reproduce = CfgNode(new_allowed=True) # seed for reproduction
    cfg.reproduce.seed = 100  # a seed for all everything

    # PLM PARAMETERS
    ##################################
    cfg.plm = CfgNode(new_allowed=True)
    cfg.plm.model_name = None # the model name, e.g. bert, roberta, gpt2, ...
                # for all the available model, please check the ./plms directory.
    cfg.plm.model_path = None
    cfg.plm.specials_to_add = ['<pad>'] # always need to add pad token, if the tokenizer doesn't have one.
    cfg.plm.optimize = CfgNode(new_allowed=True)
    cfg.plm.optimize.name = 'AdamW'  # TODO: currently not in use.
    cfg.plm.optimize.freeze_para = False
    cfg.plm.optimize.no_decay = ['bias', 'LayerNorm.weight']
    cfg.plm.optimize.lr = 0.0005
    cfg.plm.optimize.weight_decay = 0.01
    cfg.plm.optimize.betas = [0.9, 0.999]
    cfg.plm.optimize.eps = 1.0e-8
    cfg.plm.optimize.scheduler = CfgNode(new_allowed=True)
    cfg.plm.optimize.scheduler.type = None      # by default, it will choose get_linear_schedule_with_warmup
    cfg.plm.optimize.scheduler.num_warmup_steps = 500

    ## LOGIN and CHECKPOINTING ##################################################
    ## in logging, each experiment will create a separate folder for saving log.txt
    ## , (full) config.json, and the checkpoint (if use the same path).
    ## logging is in the following format：
    ## ./log
    ##  - DIR_NAME_1
    ##    - log.txt
    ##    - config.yaml
    ##    - checkpoint.pt
    ##    - ...
    ##  - DIR_NAME_2
    ##    - ...
    ##
    cfg.logging = CfgNode(new_allowed=True)
    cfg.logging.path_base = 'logs' # the path base of all the logs.
    cfg.logging.file_level = 'NOTSET' # make sure it's an option of logging package
    cfg.logging.console_level = 'INFO' # make sure it's an option of logging package
    cfg.logging.unique_string = None  # the generated (or usr defined) unique string for one experiment.
    cfg.logging.unique_string_keys = ['dataset.name', 'plm.model_path', 'template', 'verbalizer', 'datetime'] # used to generate the unique string for saving
        #- dataset.name
        #- plm.model_path # only keep the last folder name in code,
            # .i.e ../.cache/roberta-large/ will save as roberta-large
        #- template
        #- verbalizer
        #- datetime  # a 12-digit string recording the date time of running the experiment, i.e., YYMMDDHHMMSS.
    cfg.logging.datetime_format = "%m%d%H%M%S%f" # only useful when unique_string_keys includes `datetime`.
        #  make sure it's a valid format for datetime package.
    cfg.logging.path = None # always keep none to let the config generate a full path according to
            # path_base and unique_string.
    cfg.logging.overwrite = True # if a same log path exists, overwrite it.

    # CHECKPOINT
    ######################################
    cfg.checkpoint = CfgNode(new_allowed=True) # checkpoint use the same directory as logging.
    cfg.checkpoint.save_latest = True # Normally set to False to reduce memory use, set
                        # to true to allow resuming learning process.
    cfg.checkpoint.save_best = True   # Keep saving the epoch of the best-performance.
    cfg.checkpoint.higher_better = True # is the metric to determine best checkpoint higher better?


    ## PIPELINE #######################################################

    cfg.train = CfgNode(new_allowed=True)
    cfg.train.num_epochs = 5 # the number of training epochs.
    cfg.train.num_training_steps = None
    cfg.train.batch_size = 2 # the batch_size.
    cfg.train.shuffle_data = True # whether shuffle the training data.
    cfg.train.teacher_forcing = False # whether perform teacher forcing in training.
                        # if true, the desired prediction on each mask will
                        # be filled in the mask.
    cfg.train.gradient_accumulation_steps = 1 # update weight  every N step of training.
                        # set 1 to disable gradient accumulation.
    cfg.train.max_grad_norm = -1.0 # <0 for unlimited gradients norm
    cfg.train.clean = False # set to True for not saving checkpoint and no tensorboard logging

    cfg.dev = CfgNode(new_allowed=True)
    cfg.dev.batch_size = 2 # evaluationn batch_size, can be a bit larger than training batch_size
    cfg.dev.shuffle_data = False # whether to perform data shuffling in evaluation

    cfg.test = CfgNode(new_allowed=True)
    cfg.test.batch_size = 2 # evaluationn batch_size, can be a bit larger than training batch_size
    cfg.test.shuffle_data = False # whether to perform data shuffling in evaluation

    ## TASK ##########################################################@
    cfg.task = 'classification'

    cfg.classification = CfgNode(new_allowed=True)
    cfg.classification.parent_config = 'task'
    cfg.classification.metric = ['micro-f1']  # the first one will be the main  to determine checkpoint. whether the higher metric value is better.
    cfg.classification.loss_function = 'cross_entropy' # the loss function for classification

    # LMBFF-classification config ###########################################################W
    cfg.classification.auto_t = False
    cfg.classification.auto_v = False

    cfg.template_generator = CfgNode(new_allowed=True)
    cfg.template_generator.plm = CfgNode(new_allowed=True)
    cfg.template_generator.plm.model_name = 't5' # the model name, e.g. bert, roberta, gpt2, ...
                # for all the available model, please check the ./plms directory.
    cfg.template_generator.plm.model_path = None
    cfg.template_generator.plm.specials_to_add = ['<pad>'] # always need to add pad token, if the tokenizer doesn't have one.

    cfg.template_generator.max_length = 20 # maximum length of generated template
    cfg.template_generator.target_number = 2 # number of parts to generate, e.g. in T5, every <extra_id_{}> token is one part
    cfg.template_generator.beam_width = 5
    cfg.template_generator.length_limit = None # List[str] length limit for each part of content
    cfg.template_generator.template = CfgNode(new_allowed=True)
    cfg.template_generator.template.text = None
    cfg.template_generator.template.mask_token = '<mask>'
    cfg.template_generator.template.placeholder_mapping = CfgNode(new_allowed=True)
    cfg.template_generator.template.placeholder_mapping['<text_a>'] = 'text_a'
    cfg.template_generator.template.placeholder_mapping['<text_b>'] = 'text_b'
    cfg.template_generator.template.file_path = None
    cfg.template_generator.template.choice = 0

    # verbalizer_generator, refer to https://arxiv.org/abs/2010.13641
    cfg.verbalizer_generator = CfgNode(new_allowed=True)
    cfg.verbalizer_generator.candidate_num = 1 # the number of candidates for further selection
    cfg.verbalizer_generator.label_word_num_per_class = 1
    cfg.verbalizer_generator.score_fct = 'llr' # the scoring function of label words selection. ``llr'' means log likelihood ratio, corresponding to Equation (7); ``ce'' means cross entropy, corresponding to Equation (6). As the paper points out, ``llr'' is significantly better than 'ce', we only keep it to matchthe original code.
    cfg.verbalizer_generator.normalize = True # whether to perform normalization of unbalanced training dataset, as Equation (5)



    cfg.generation = CfgNode(new_allowed=True) # Adding any arguments for generation here.
    cfg.generation.parent_config = 'task'
    cfg.generation.metric = ['sentence_bleu']
    cfg.generation.max_length = 512   # the max_length of the generated sentence. INCLUDING the input_ids. So: generation.max_length > dataloader.max_seq_length
    cfg.generation.max_new_tokens = None
    cfg.generation.min_length = 5
    cfg.generation.temperature = 1.0
    cfg.generation.do_sample = False
    cfg.generation.top_k = 0
    cfg.generation.top_p = 0.9
    cfg.generation.repetition_penalty = 1.0 ##args.repetition_penalty,
    cfg.generation.num_beams = 5
    cfg.generation.bad_words_ids = [[628, 198]]


    cfg.relation_classification = CfgNode(new_allowed=True)
    cfg.relation_classification.parent_config = 'task'

    ## DATASET #########################################################
    cfg.dataset = CfgNode(new_allowed=True)
    cfg.dataset.name = None   # the name of the dataset, for the supported choices,
            # please see the processors in ./data_utils/
    cfg.dataset.path = None  # whether is the dataset saved in your local machine.
    cfg.dataset.label_path_sep = None # label path separation token, only for hierarchical label

    ## DATALOADER ######################################################
    cfg.dataloader = CfgNode(new_allowed=True)
    cfg.dataloader.max_seq_length = 256 # max_seq_length
    cfg.dataloader.decoder_max_length = 256 # the decoder max length to truncate decoder input sequence
                        # if it is an encoder-decoder architecture. Note that it's not equavalent
                        # to generation.max_length which is used merely in the generation phase.
    cfg.dataloader.truncate_method = "head" # choosing from balanced, head, tail

    ## LEARINING SETTING  ####################################################
    cfg.learning_setting = None   # selecting from "full", "zero-shot", "few-shot"

    cfg.zero_shot = CfgNode(new_allowed=True)
    cfg.zero_shot.parent_config = 'learning_setting'

    cfg.few_shot = CfgNode(new_allowed=True)
    cfg.few_shot.parent_config = 'learning_setting'
    cfg.few_shot.few_shot_sampling = None

    cfg.sampling_from_train = CfgNode(new_allowed=True)
    cfg.sampling_from_train.parent_config = 'few_shot_sampling'
    cfg.sampling_from_train.num_examples_per_label = 10
    cfg.sampling_from_train.also_sample_dev = True
    cfg.sampling_from_train.num_examples_per_label_dev = 10
    cfg.sampling_from_train.seed = [123]

    ## CALIBRATION ###########################################################
    cfg.calibrate = None # leave blank to use no calibrate

    cfg.contextualized_calibrate = CfgNode(new_allowed=True)
    cfg.contextualized_calibrate.parent_config = 'calibrate'
    cfg.contextualized_calibrate.num_example = None
    cfg.contextualized_calibrate.use_split = 'train'

    cfg.pmi_calibrate = CfgNode(new_allowed=True)
    cfg.pmi_calibrate.parent_config = 'calibrate'

    ## PROMPT SPECIFIC CONFIG ############################################
    cfg.template = None
    cfg.verbalizer = None

    cfg.manual_template = CfgNode(new_allowed=True)
    cfg.manual_template.parent_config = 'template'
    cfg.manual_template.text = None
    cfg.manual_template.mask_token = '<mask>'
    cfg.manual_template.placeholder_mapping = CfgNode(new_allowed=True)
    cfg.manual_template.placeholder_mapping['<text_a>'] = 'text_a'
    cfg.manual_template.placeholder_mapping['<text_b>'] = 'text_b'
    cfg.manual_template.file_path = None
    cfg.manual_template.choice = 0
    cfg.manual_template.optimize = None  # the parameters related to optimize the template


    cfg.automatic_verbalizer = CfgNode(new_allowed=True)
    cfg.automatic_verbalizer.parent_config = 'verbalizer'
    cfg.automatic_verbalizer.num_cadidates = 1000
    cfg.automatic_verbalizer.label_word_num_per_class = 1
    cfg.automatic_verbalizer.num_searches = 1
    cfg.automatic_verbalizer.score_fct = 'llr'
    cfg.automatic_verbalizer.balance = True
    cfg.automatic_verbalizer.optimize = CfgNode(new_allowed=True)
    cfg.automatic_verbalizer.optimize.level = 'epoch'
    cfg.automatic_verbalizer.num_classes = None
    cfg.automatic_verbalizer.init_using_split = 'valid'

    cfg.one2one_verbalizer = CfgNode(new_allowed=True)
    cfg.one2one_verbalizer.parent_config = 'verbalizer'
    cfg.one2one_verbalizer.label_words = None
    cfg.one2one_verbalizer.prefix = " "
    cfg.one2one_verbalizer.multi_token_handler = 'first'
    cfg.one2one_verbalizer.file_path = None
    cfg.one2one_verbalizer.choice = None
    cfg.one2one_verbalizer.num_classes = None
    cfg.one2one_verbalizer.optimize = None

    cfg.manual_verbalizer = CfgNode(new_allowed=True)
    cfg.manual_verbalizer.parent_config = 'verbalizer'
    cfg.manual_verbalizer.label_words = None
    cfg.manual_verbalizer.prefix = " "
    cfg.manual_verbalizer.multi_token_handler = 'first'
    cfg.manual_verbalizer.file_path = None
    cfg.manual_verbalizer.choice = None
    cfg.manual_verbalizer.num_classes = None
    cfg.manual_verbalizer.optimize = None

    cfg.prefix_tuning_template = CfgNode(new_allowed=True)
    cfg.prefix_tuning_template.parent_config = 'template'
    cfg.prefix_tuning_template.text = None
    cfg.prefix_tuning_template.mask_token = '<mask>'
    cfg.prefix_tuning_template.num_token = 5
    cfg.prefix_tuning_template.placeholder_mapping = CfgNode(new_allowed=True)
    cfg.prefix_tuning_template.placeholder_mapping['<text_a>'] = 'text_a'
    cfg.prefix_tuning_template.placeholder_mapping['<text_b>'] = 'text_b'
    cfg.prefix_tuning_template.prefix_dropout = 0.0
    cfg.prefix_tuning_template.mid_dim = 512
    cfg.prefix_tuning_template.optimize = CfgNode(new_allowed=True)
    cfg.prefix_tuning_template.optimize.name = 'AdamW'
    cfg.prefix_tuning_template.optimize.lr = 0.00005
    cfg.prefix_tuning_template.optimize.betas = [0.9, 0.999]
    cfg.prefix_tuning_template.optimize.adam_epsilon = 1.0e-8
    cfg.prefix_tuning_template.optimize.weight_decay = 0.0
    cfg.prefix_tuning_template.optimize.no_decay = ['bias', 'LayerNorm.weight']
    cfg.prefix_tuning_template.optimize.scheduler = CfgNode(new_allowed=True)
    cfg.prefix_tuning_template.optimize.scheduler.num_warmup_steps = 0

    cfg.mixed_template = CfgNode(new_allowed=True)
    cfg.mixed_template.parent_config = 'template'
    cfg.mixed_template.text = None
    cfg.mixed_template.mask_token = '<mask>'
    cfg.mixed_template.placeholder_mapping = CfgNode(new_allowed=True)
    cfg.mixed_template.placeholder_mapping['<text_a>'] = 'text_a'
    cfg.mixed_template.placeholder_mapping['<text_b>'] = 'text_b'
    cfg.mixed_template.optimize = CfgNode(new_allowed=True)
    cfg.mixed_template.optimize.name = 'AdamW'
    cfg.mixed_template.optimize.lr = 0.00005
    cfg.mixed_template.optimize.betas = [0.9, 0.999]
    cfg.mixed_template.optimize.adam_epsilon = 1.0e-8
    cfg.mixed_template.optimize.weight_decay = 0.0
    cfg.mixed_template.optimize.no_decay = ['bias', 'LayerNorm.weight']
    cfg.mixed_template.optimize.scheduler = CfgNode(new_allowed=True)
    cfg.mixed_template.optimize.scheduler.num_warmup_steps = 0


    return cfg

