from yacs.config import CfgNode
from .typing_dataset import PROCESSORS as TYPING_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .fewglue_dataset import PROCESSORS as SUPERGLUE_PROCESSORS
from .relation_classification_dataset import PROCESSORS as RC_PROCESSORS
from .lama_dataset import PROCESSORS as LAMA_PROCESSORS
from .conditional_generation_dataset import PROCESSORS as CG_PROCESSORS
from .utils import InputExample, InputFeatures
from .data_sampler import FewShotSampler
# support loading transformers datasets from https://huggingface.co/docs/datasets/
from .nli_dataset import PROCESSORS as NLI_PROCESSORS

from openprompt.utils.logging import logger
from openprompt.data_utils.huggingface_dataset import PROCESSORS as HF_PROCESSORS

PROCESSORS = {
    **TYPING_PROCESSORS,
    **TC_PROCESSORS,
    **SUPERGLUE_PROCESSORS,
    **RC_PROCESSORS,
    **LAMA_PROCESSORS,
    **CG_PROCESSORS,
    **LAMA_PROCESSORS,
    **HF_PROCESSORS,
    **NLI_PROCESSORS,
}


def load_dataset(config: CfgNode, return_class=True, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset

    processor = PROCESSORS[dataset_config.name.lower()]()

    train_dataset = None
    valid_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {dataset_config.path}.")
        try:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {dataset_config.path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(dataset_config.path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {dataset_config.path}.")
    # checking whether downloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset

