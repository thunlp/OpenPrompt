## load数据，对每一个batch包装template，返回的和普通data_loader一样
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from openprompt.plms import get_tokenizer_wrapper
from yacs.config import CfgNode
from openprompt.utils.logging import logger



class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer. 
    
    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper. 
    """
    def __init__(self, 
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset
        
        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_class = get_tokenizer_wrapper(tokenizer)
        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
                        "max_seq_length":max_seq_length,
                        "truncate_method":truncate_method,
                        "decoder_max_length":decoder_max_length,
                        "predict_eos_token":predict_eos_token,
                        "tokenizer": tokenizer,
                        **kwargs,
                        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
        

        self.tokenizer_wrapper = get_tokenizer_wrapper(tokenizer)(**to_pass_kwargs)
        
        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"
        
        # processs
        self.wrap()
        self.tokenize()

        def prompt_collate_fct(batch: List[Union[Dict, InputFeatures]]):
            r'''
            This function is used to collate the current prompt.

            Args:
                batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

            Returns:
                :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
            '''

            
            elem = batch[0]
            return_dict = {key: default_collate([d[key] for d in batch]) for key in elem}
            return InputFeatures(**return_dict)

        self.dataloader = DataLoader(self.tensor_dataset, 
                                     batch_size = self.batch_size,
                                     shuffle = self.shuffle,
                                     collate_fn = prompt_collate_fct
                                    )
    
    
    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List): # TODO change to iterable 
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError
    
    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer, 
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)
        


    def __getitem__(self, idx):
        r"""simulate the ``torch.utils.data.Dataset``'s behavior.
        """
        return self.tensor_dataset[idx]
    
    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``, 
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
    '''
    def __init__(self,
                 model: PreTrainedModel, 
                 template: Template,
                 ):
        super().__init__()
        self.model = model
        self.template = template

        # get model's forward function's keywords
        self.forward_keys = signature(self.model.forward).args
        
    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs =  self.model(**input_batch)
        return outputs
    
    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

    


class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a PromptModel) into the
    logits of the labels, using a verbalizer. 

    Args:
        model (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
    '''
    def __init__(self,
                 model: PreTrainedModel, 
                 template: Template,
                 verbalizer: Verbalizer,
                 ):
        super().__init__()
        self.model = model
        self.template = template
        self.prompt_model = PromptModel(model, template)
        self.verbalizer = verbalizer


    @property
    def device(self,):
        r"""
        Register the device parameter.
        """
        return self.model.device

    def extract_logits(self,
                       logits: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get logits of all <mask> token
        Project the logits of shape
        (batch_size, max_seq_length, vocab_size)
        into logits of shape (if num_mask_token > 1)
        (batch_size, num_mask_token, vocab_size)
        or into logits of shape (if num_mask_token = 1)
        (batch_size, vocab_size).

        Args:
            logits (:obj:`torch.Tensor`): The original logits of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch
        
        Returns:
            :obj:`torch.Tensor`: The extracted logits of ``<mask>`` tokens.
            
        """
        logits = logits[torch.where(batch['loss_ids']>0)]
        logits = logits.view(batch['loss_ids'].shape[0], -1, logits.shape[1])
        if logits.shape[1] == 1:
            logits = logits.view(logits.shape[0], logits.shape[2])
        return logits
        
    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" keys in batch: 
        """
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits = self.extract_logits(logits, batch)
        label_words_logits = self.verbalizer.process_logits(logits=logits, batch=batch)

        if 'label' in batch:
            pass #TODO add caculate label loss here

        return label_words_logits
    
    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits = self.extract_logits(logits, batch)
        return logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer
    
    def state_dict(self):
        r""" Save the model using template and verbalizer's save methods.
        Args:
            path (:obj:`str`): the full path of the checkpoint.
            save_plm (:obj:`bool`): whether saving the pretrained language model.
            kwargs: other information, such as the achieved metric value. 
        """
        _state_dict = {}
        _state_dict['plm'] = self.model.state_dict()
        _state_dict['template'] = self.template.state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        if 'plm' in state_dict:
            self.model.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])






class PromptForGeneration(nn.Module, GenerationMixin):
    r'''``PromptModel`` with generation loss caculation and generation utils integrated.


    Args:
        model (:obj:`PretrainedModel`): A pre-traiend model you decide to use for generation, e.g. GPT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``PrefixTemplate``.
        tokenizer (:obj:`Tokenizer`): A ``Tokenizer`` of the current model.
        gen_config (:obj:`CfgNode`): The generation configs to pass into `GenerationMixin.generate <https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate>`_
    '''

    def __init__(self,
                 model: PreTrainedModel, 
                 template: Template,
                 gen_config: CfgNode,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
                 
        super().__init__()
        self.model = model
        self.template = template
        if tokenizer is None:
            assert self.template.tokenizer is not None, "Tokenizer can't be set from input args or template"
            self.tokenizer = template.tokenizer
        else:
            self.tokenizer = tokenizer
        self.prompt_model = PromptModel(model, template)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.config = model.config
        for key in gen_config:
            setattr(self.config, key, gen_config[key])
        self.in_generation_function = False

    @property
    def device(self):
        return self.model.device

    def shift_logits_and_labels(self, 
                                logits, 
                                batch: InputFeatures):

        r"""
        Left shift the label, and make label of the positions that are
        not loss position to -100, which is the ignore index in pytorch's
        loss function.

        Args:
            logits (:obj:`torch.Tensor`):
            batch (:obj:InputFeatures): The input features of batchified data sequences.
        
        Returns:
            shift_logits (:obj:`torch.Tensor`):
            shift_input_ids (:obj:`List[int]`):

        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = batch['loss_ids'][..., 1:].contiguous()
        if self.config.is_encoder_decoder:
            shift_input_ids = batch['decoder_input_ids'][..., 1:].contiguous()
        else:
            shift_input_ids = batch['input_ids'][..., 1:].contiguous()
        shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    def forward(self, *args, **kwargs):
        r"""In generation process, it will use the plm's forward function.
        This is because, in the first step we will directly call the process_batch function to 
        generate initial input with the template, after that the all template
        have been processed into the past_key_value,
        then we can use the normal generation function. 
        In learning process, the forward is linked to ``_forward`` functions.
        in which the loss will be calcated for all the postions in the same time. 
        """
        if self.in_generation_function:
            return self.prompt_model.model.forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is the forward method of the training of generation in prompt-learning framework. 
        
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        
        Returns:
            loss(:obj:torch.Tensor): The loss of the current generation procedure.
        """
        
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits, labels = self.shift_logits_and_labels(logits, batch)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1) #TODO support more objectives
        loss = loss.mean()
        return loss
    
    
    def generate(self, batch: Union[Dict, InputFeatures], **generation_kwargs):
        r""" This function wraps the generate() methods in parent class ``GenerationMixin``.
        Forward uses the ``PretrainedModel``'s forward method. 
        generation_kwargs include all the parameters that are passed in to 
        ``transformers.generation_util.GenerationMixin.generate``
    
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        
        Returns:
            output_sequences (:obj:List[torch.Tensor]): The raw sequences generated by the generation model.
            generated_sentences (:obj:List[torch.Tensor]): The generated sentences that have been post-processed.
        """
        self.generate_ith_token = 0
        if self.config.is_encoder_decoder:
            loss_ids_start = batch['loss_ids'].argmax(dim=-1)
            assert loss_ids_start.min() == loss_ids_start.max(), "The generation start from different position in a batch."
            batch['decoder_input_ids'] = batch['decoder_input_ids'][:, :loss_ids_start.min()+1]
            input_length = batch['decoder_input_ids'].size(1)
        else:
            input_length = batch['input_ids'].size(1)
        input_generation_kwargs = {key: value for key,value in generation_kwargs.items() if key in signature(GenerationMixin.generate).args}
        self.in_generation_function = True
        output_sequences = super().generate(**batch, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        self.in_generation_function = False
        generated_sentences = self.post_processing(output_sequences=output_sequences, input_length=input_length)
        return output_sequences, generated_sentences
    
    def post_processing(self, output_sequences, input_length):
        r"""
            Post-process the sequences generated by the generation model.

            Args:
                output_sequences (:obj:`torch.Tensor`): The raw sequences generated by the generation model.
                input_length (:obj:`int`): The length of the input sequence.
            
            Returns:
                :obj:`List`: The generated sentences that have been post-processed.
        """
        output_sequences = output_sequences.cpu().tolist()
        generated_sentences = []
        for seq in output_sequences:
            # Decode text
            seq = seq[input_length:]
            text_output = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            idx = text_output.find(self.tokenizer.eos_token)
            if idx >= 0:
                text_output = text_output[:idx]
            text_output = text_output.strip()
            generated_sentences.append(text_output)
        return generated_sentences


    
    def prepare_inputs_for_generation(self, input_ids: Optional[torch.Tensor] = None,
                                         **model_kwargs):
        r"""When the `past` not in model_kwargs, we prepare the input from scratch. 
        When `past` is in model_kwargs, we don't need to prepare the template wrapped input,
        instead we use the inner pretrain_models' function to prepare the next step's input.
        `model_kwargs` includes all the argument passed in the `batch`: InputFeatures, except `input_ids`
        , as long as they do not conflict with keywords in ``generation_kwargs``.    if 'past' not in model_kwargs: # the past_key_value not in model_kwargs, then we need to prepare input from scrath
        , as long as they do not conflict with keywords in ``generation_kwargs''.

        Args:
            input_ids(:obj:`torch.Tensor`): Indices of input sequence tokens in the vocabulary.
        """
 
        
        if self.generate_ith_token == 0 and 'encoder_outputs' not in model_kwargs: # generating the first token in decoder only setting.

            batch = InputFeatures(input_ids=input_ids, **model_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
        else: # generating the subsequence generation can use the default setting
            model_inputs = self.prompt_model.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        self.last_model_inputs = model_inputs  # to update the model_kwargs in _update_model_kwargs_for_generation, in-place operation.
        return model_inputs
    
    
    def _update_model_kwargs_for_generation(self,
        outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        r""" The parents class's ``_update_model_kwargs_for_generation`` method will
        add past_key_values to model_kwargs, and update ``token_type_ids``, and ``attention_mask_ids``.

        In case some of the model_kwargs are modified in the prepare_inputs_for_generation function
        and should be used as the subsequent model_kwargs, we upate these kwargs before the parent class
        call. 

        Other updates should be added here after the parent's function call.

        Args:
            outputs (:obj:`torch.Tensor`): 
            is_encoder_decoder (:obj:`bool`, defaults to False): 
        """
        if self.generate_ith_token == 0:
            for key in self.last_model_inputs:
                if key in model_kwargs:
                    model_kwargs[key] = self.last_model_inputs[key]
        
        model_kwargs = super(PromptForGeneration, PromptForGeneration)._update_model_kwargs_for_generation(outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder)
        self.generate_ith_token += 1
        return model_kwargs


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.prompt_model.model.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            batch = InputFeatures(input_ids=input_ids, **encoder_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
            model_kwargs["encoder_outputs"] = encoder(return_dict=True, **model_inputs)
        return model_kwargs
    
    def state_dict(self):
        r""" Save the model using template and verbalizer's save methods.
        Args:
            path (:obj:`str`): the full path of the checkpoint.
            save_plm (:obj:`bool`): whether saving the pretrained language model.
            kwargs: other information, such as the achieved metric value. 
        """
        _state_dict = {}
        _state_dict['plm'] = self.model.state_dict()
        _state_dict['template'] = self.template.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        if 'plm' in state_dict:
            self.model.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
    
    def _reorder_cache(self, past, beam_idx):
        r"""Use the plm's default _reorder_cache function
        """
        return self.model._reorder_cache(past, beam_idx)