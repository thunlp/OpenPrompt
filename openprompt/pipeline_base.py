from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
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
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup



class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the original dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to truncate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # process
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wrapped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        model = self.plm
        if hasattr(model, "encoder") and hasattr(model.encoder, "main_input_name"):
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, "main_input_name", "input_ids")
        self.main_input_name = main_input_name

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the label words (obtained by the current verbalizer).
        """
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits

    def predict(self):
        pass

    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template, plm and verbalizer's save methods."""
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template, plm and verbalizer's load methods."""
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    #     self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")


class PromptForGeneration(nn.Module, GenerationMixin):
    r'''``PromptModel`` with generation loss calculation and generation utils integrated.


    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for generation, e.g. GPT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``PrefixTuningTemplate``.
        tokenizer (:obj:`Tokenizer`): A ``Tokenizer`` of the current model.
        gen_config (:obj:`CfgNode`): The generation configs to pass into `GenerationMixin.generate <https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate>`_
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''

    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 gen_config: Optional[CfgNode] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
        super().__init__()
        self.freeze_plm = freeze_plm
        if tokenizer is None:
            assert template.tokenizer is not None, "Tokenizer can't be set from input args or template"
            self.tokenizer = template.tokenizer
        else:
            self.tokenizer = tokenizer
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.config = plm.config
        if gen_config:
            for key in gen_config:
                setattr(self.config, key, gen_config[key])
        self.in_generation_function = False

        self.main_input_name = self.prompt_model.main_input_name # for transformers 4.17.0 and higher.

    @property
    def generation_config(self):
        return self.plm.generation_config
    
    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        return self.plm.device
    
    def can_generate(self):
        if hasattr(self.prompt_model.plm, "can_generate"):
            return self.prompt_model.plm.can_generate
        else:
            raise AttributeError("the corresponding plm does not have `can_generate` attribute")

    def shift_logits_and_labels(self,
                                logits,
                                loss_ids,
                                reference_ids):

        r"""
        Left shift the label, and make label of the positions that are
        not loss position to -100, which is the ignore index in pytorch's
        loss function.

        Args:
            logits (:obj:`torch.Tensor`):
            batch (:obj:`InputFeatures`): The input features of batchified data sequences.

        Returns:
            shift_logits (:obj:`torch.Tensor`):
            shift_input_ids (:obj:`List[int]`):

        """

        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    def forward(self, *args, **kwargs):
        r"""In generation process, it will use the plm's forward function.
        This is because, in the first step we will directly call the process_batch function to
        generate initial input with the template, after that the all template
        have been processed into the past_key_value,
        then we can use the normal generation function.
        In learning process, the forward is linked to ``_forward`` functions.
        in which the loss will be calculated for all the positions in the same time.
        """
        if self.in_generation_function:
            return self.plm.forward(*args, **kwargs)
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
        if self.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']  # in case in some template, these field is dropped
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1) # TODO support more objectives
        loss = loss.mean()
        return loss

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """skip model kwargs validation, as all arguments will be filtered before feeding into plm in PromptModel"""
        return


    def generate(self, batch: Union[Dict, InputFeatures], verbose: Optional[bool]=False, **generation_kwargs):
        r""" This function wraps the generate() methods in parent class ``GenerationMixin``.
        Forward uses the ``PretrainedModel``'s forward method.
        generation_kwargs include all the parameters that are passed in to
        ``transformers.generation_util.GenerationMixin.generate``

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
            verbose (:obj:`Optional[bool]`): Set to true to verbose the generated sentence.

        Returns:
            output_sequences (:obj:`List[torch.Tensor]`): The raw sequences generated by the generation model.
            generated_sentences (:obj:`List[torch.Tensor]`): The generated sentences that have been post-processed.
        """
        input_generation_kwargs = generation_kwargs
        if self.config.is_encoder_decoder:
            loss_ids_start = batch['loss_ids'].argmax(dim=-1)
            assert loss_ids_start.min() == loss_ids_start.max(), "The generation start from different position in a batch."
            batch['decoder_input_ids'] = batch['decoder_input_ids'][:, :loss_ids_start.min()+1]
            input_length = batch['decoder_input_ids'].size(1)
            batch_size = batch['decoder_input_ids'].size(0)

            self.generate_ith_token = 0
            self.in_generation_function = True
            output_sequences = super().generate(**batch, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            self.in_generation_function = False
            output_sequences = output_sequences.cpu().tolist()
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_length)
        else:
            input_length = batch['input_ids'].size(1)
            batch_size = batch['input_ids'].size(0)

            # Currently huggingface transformers only support single sample generation, or padding to the left (instead of the right).
            # because it will only extract the last position of the output
            # generate one_by_one
            if 'input_ids_len' in batch:
                input_real_lens = batch['input_ids_len']
            else:
                input_real_lens = torch.sum((batch['input_ids'] != self.tokenizer.pad_token_id).to(torch.int), dim=-1)
            output_sequences = []
            for instance_id in range(batch_size):
                # remove the pad token
                instance = {key: batch[key][instance_id:instance_id+1][:,:input_real_lens[instance_id]] for key in batch if isinstance(batch[key], torch.Tensor) and batch[key].shape[:2]==torch.Size([batch_size, input_length])}
                self.generate_ith_token = 0
                self.in_generation_function = True
                output_sequence = super().generate(**instance, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                self.in_generation_function = False
                output_sequences.extend(output_sequence.cpu().tolist()) # TODO: to support generate multiple sentence
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_real_lens.cpu().tolist())
        if verbose:
            logger.info(f"Generated:{generated_sentences}")
        return output_sequences, generated_sentences



    def post_processing(self, output_sequences, input_lengths):
        r"""
            Post-process the sequences generated by the generation model.

            Args:
                output_sequences (:obj:`torch.Tensor`): The raw sequences generated by the generation model.
                input_lengths (:obj:`int` or `list`): The length(s) of the input sequence.

            Returns:
                :obj:`List`: The generated sentences that have been post-processed.
        """
        generated_sentences = []
        if type(input_lengths)==int:
            input_lengths = [input_lengths] * len(output_sequences)
        for sent_id, seq in enumerate(output_sequences):
            seq = seq[input_lengths[sent_id]:]

            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=False)
                idx = text_output.find(self.tokenizer.eos_token)
                if idx >= 0:
                    text_output = text_output[:idx]
            else:
                text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            text_output = text_output.strip()
            generated_sentences.append(text_output)
        return generated_sentences



    def prepare_inputs_for_generation(self, input_ids: Optional[torch.Tensor] = None,
                                         **model_kwargs):
        r"""This function wraps the ``prepare_inputs_for_generation`` function in the huggingface transformers.

        When the `past` not in model_kwargs, we prepare the input from scratch.
        When `past` is in model_kwargs, we don't need to prepare the template wrapped input,
        instead we use the inner pretrain_models' function to prepare the next step's input.
        `model_kwargs` includes all the argument passed in the `batch`: InputFeatures, except ``input_ids``
        , as long as they do not conflict with keywords in ``generation_kwargs``.    if 'past' not in model_kwargs: # the past_key_value not in model_kwargs, then we need to prepare input from scrath
        , as long as they do not conflict with keywords in ``generation_kwargs``.

        Args:
            input_ids(:obj:`torch.Tensor`): Indices of input sequence tokens in the vocabulary.
        """
        if self.generate_ith_token == 0 and 'encoder_outputs' not in model_kwargs: # generating the first token in decoder only setting.

            batch = InputFeatures(input_ids=input_ids, **model_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
            # check the compatibility for more models. Having checked gpt2, T5
        else: # generating the subsequence generation can use the default setting
            model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)
        self.last_model_inputs = model_inputs  # to update the model_kwargs in _update_model_kwargs_for_generation, in-place operation.
        return model_inputs


    def _update_model_kwargs_for_generation(self,
        outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        r""" The parents class's ``_update_model_kwargs_for_generation`` method will
        add ``past_key_values`` to model_kwargs, and update ``token_type_ids``, and ``attention_mask_ids``.

        In case some of the model_kwargs are modified in the prepare_inputs_for_generation function
        and should be used as the subsequent model_kwargs, we update these kwargs before the parent class
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
        model_kwargs = super()._update_model_kwargs_for_generation(outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder)
        self.generate_ith_token += 1
        return model_kwargs


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs, model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        r""" This function resemble the function in GeneraionMix

        Args:
            input_ids (:obj:`torch.LongTensor`) The input ids for
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.plm.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn") or argument.startswith("use_cache"))
            }
            model_input_name = model_input_name if model_input_name is not None else self.main_input_name
            batch = {model_input_name:input_ids, **encoder_kwargs}
            model_inputs = self.prompt_model.prepare_model_inputs(batch) # This line differs from the orinigal code base, we should process the input
            # with our template, then pass it into the model.
            # some of the arguments may have been changed by the template,
            # e.g. the attention mask. Here we update the model_kwargs
            for key in model_kwargs:
                if key in model_inputs:
                    model_kwargs[key] = model_inputs[key]
            model_kwargs["encoder_outputs"] = encoder(return_dict=True, **model_inputs)
        return model_kwargs

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template and plm's save methods. """
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template and plm's load methods. """
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        r"""Use the plm's default _reorder_cache function
        """
        return self.plm._reorder_cache(past, beam_idx)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")
