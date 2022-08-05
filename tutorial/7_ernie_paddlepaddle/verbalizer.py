import paddle
from abc import abstractmethod
from typing import *
import numpy as np
import paddle.nn.functional as F

class Verbalizer(paddle.nn.Layer):
    r'''
    Base class for all the verbalizers.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
    '''
    def __init__(self,
                 tokenizer = None,
                 classes= None,
                 num_classes: Optional[int] = None,
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None
            # raise AttributeError("No able to configure num_classes")
        self._in_on_label_words_set = False

    @property
    def label_words(self,):
  
        if not hasattr(self, "_label_words"):
            raise RuntimeError("label words haven't been set.")
        return self._label_words

    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:
            return
        self._label_words = self._match_label_words_to_label_ids(label_words)
        if not self._in_on_label_words_set:
            self.safe_on_label_words_set()
        # else:
        #     logger.warning("Reset label words in on_label_words_set function. Is this intended?")

    def _match_label_words_to_label_ids(self, label_words): # TODO newly add function after docs written # TODO rename this function
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError("name of classes in verbalizer are different from those of dataset")
            label_words = [ # sort the dict to match dataset
                label_words[c]
                for c in self.classes
            ] # length: label_size of the whole task
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
            # logger.info("""
            # Your given label words is a list, by default, the ith label word in the list will match class i of the dataset.
            # Please make sure that they have the same order.
            # Or you can pass label words as a dict, mapping from class names to label words.
            # """)
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return label_words

    def safe_on_label_words_set(self,):
        self._in_on_label_words_set = True
        self.on_label_words_set()
        self._in_on_label_words_set = False

    def on_label_words_set(self,):
        r"""A hook to do something when textual label words were set.
        """
        pass

    @property
    def vocab(self,) -> Dict:
        if not hasattr(self, '_vocab'):
            self._vocab = self.tokenizer.convert_ids_to_tokens(np.arange(self.vocab_size).tolist())
        return self._vocab

    @property
    def vocab_size(self,) -> int:
        return self.tokenizer.vocab_size

    @abstractmethod
    def generate_parameters(self, **kwargs) -> List:

        raise NotImplementedError

    def register_calibrate_logits(self, logits):
        r"""
        This function aims to register logits that need to be calibrated, and detach the original logits from the current graph.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits

    def process_outputs(self,
                       outputs,
                       batch,
                       **kwargs):
  

        return self.process_logits(outputs, batch=batch, **kwargs)


    def handle_multi_token(self, label_words_logits, mask):
      
        if self.multi_token_handler == "first":
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            label_words_logits = label_words_logits - 1000*(1-mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == "mean":
            label_words_logits = (label_words_logits*mask.unsqueeze(0)).sum(dim=-1)/(mask.unsqueeze(0).sum(dim=-1)+1e-15)
        else:
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits

class ErnieManualVerbalizer(Verbalizer):

    def __init__(self,
                 tokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

         # TODO should Verbalizer base class has label_words property and setter?
         # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def calibrate(self, label_words_probs, **kwargs):
     
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs
    def project(self,
            logits,
            **kwargs,
            ):
        label1 = self.label_words_ids[0]
        label2 = self.label_words_ids[1]
        tmp = []
        tmp.append(logits.index_select(label1,axis=-1))
        tmp.append(logits.index_select(label2,axis=-1))
        label_words_logits = paddle.concat(tmp,axis=1).reshape([logits.shape[0],self.label_words_ids.shape[0],self.label_words_ids.shape[1]])
#         label_words_logits = logits[:, self.label_words_ids]
#         tmp3 = []
#         for index in range(logits.shape[0]):
#             tmp2 = []
#             for j in range(self.label_words_ids.shape[0]):
#                 tmp = []
#                 for i in range(self.label_words_ids.shape[1]):
#                     tmp.append(logits.slice(axes=[0,1],starts=[index,self.label_words_ids[j][i]],ends=[index+1,(self.label_words_ids+1)[j][i]]))
#                 tmp2.append(paddle.concat(tmp))
#             tmp3.append(tmp2)
#         label_words_logits = paddle.to_tensor(tmp3)
#         label_words_logits = label_words_logits.reshape(label_words_logits.shape[:-1])
#         label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits
    def gather_outputs(self,outputs):
        return outputs[0]
    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False,return_token_type_ids = False)['input_ids']
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = paddle.zeros([max_num_label_words, max_len])
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        
        
        
        words_ids_tensor = paddle.to_tensor(words_ids)
        words_ids_mask = paddle.to_tensor(words_ids_mask)
        self.label_words_ids = words_ids_tensor
        self.words_ids_mask = words_ids_mask
        self.label_words_mask = paddle.clip(self.words_ids_mask.sum(axis=-1),max=1)
        

    
    def aggregate(self, label_words_logits):
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`paddle.Tensor`): The logits of the label words.

        Returns:
            :obj:`paddle.Tensor`: The aggregated logits from the label words.
        """
        tmp = self.label_words_mask
        label_words_logits = (label_words_logits * tmp).sum(-1)/tmp.sum(-1)
        return label_words_logits

    def process_logits(self, logits, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`paddle.Tensor`): The original logits.

        Returns:
            (:obj:`paddle.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = paddle.log(label_words_probs+1e-15)
            label_words_logits = label_words_logits

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits):

        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape([batch_size, -1]), axis=-1).reshape(logits.shape)