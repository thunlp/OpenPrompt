

from typing import *
from tqdm import tqdm
from collections import defaultdict
import warnings
import numpy as np
from paddlenlp.data import Stack, Tuple, Pad
import paddle
from utils import signature
from data_utils import InputFeatures

def create_dataloader(dataset_origin,
                  mode='train',
                  batch_size=1,
                  batchify_fn=None,
                  trans_fn=None):
    if trans_fn:
        dataset = dataset_origin.map(trans_fn)
    else:
        dataset = dataset_origin
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)
    
batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0),  # input_ids
#             Stack(), # input_embeds
            Pad(axis=0),  # attention_mask
#             Pad(axis=0), # token_type_ids
            Stack(), # label
#             Stack(),  # decoder_input_ids
#             Stack(),  # decoder_inputs_embeds
#             Stack(),  # soft_token_ids
#             Stack(),  # past_key_values
            Pad(axis = 0),  # loss_ids
#             Stack(), # guid
#             Stack(), # tgt_text
#             Stack(), # encoded_tgt_text
#             Stack(), #input_ids_len
        ): [data for data in fn(samples)]


class ErniePromptDataLoader():
    def __init__(self,
             dataset,
             template,
             tokenizer_wrapper = None,
             tokenizer = None,
             tokenizer_wrapper_class = None,
             verbalizer = None,
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

        self.dataloader = create_dataloader(dataset_origin=self.tensor_dataset,batch_size = batch_size,batchify_fn = batchify_fn)
    
    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError
    
    def tokenize(self):
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1])
            tmp = []
            for key in inputfeatures.tensorable_keys:
                value = getattr(inputfeatures, key)
                if value is not None:
                    tmp.append(paddle.to_tensor(value))
#                     setattr(inputfeatures, key, paddle.to_tensor(value))
            self.tensor_dataset.append(tmp)
    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()