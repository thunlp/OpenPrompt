from yacs.config import CfgNode
from openprompt.utils.utils import signature
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer

from transformers.utils.dummy_pt_objects import PreTrainedModel
from .manual_template import ManualTemplate
from .manual_verbalizer import ManualVerbalizer
from .mixed_template import MixedTemplate
from .one2one_verbalizer import One2oneVerbalizer
from .automatic_verbalizer import AutomaticVerbalizer
from .prefix_tuning_template import PrefixTuningTemplate
from .knowledgeable_verbalizer import KnowledgeableVerbalizer
from .ptuning_prompts import PtuningTemplate
from .ptr_prompts import PTRTemplate, PTRVerbalizer
from .knowledgeable_verbalizer import KnowledgeableVerbalizer
from .prefix_tuning_template import PrefixTuningTemplate
from .soft_template import SoftTemplate

from .prompt_generator import T5TemplateGenerator, TemplateGenerator, VerbalizerGenerator, RobertaVerbalizerGenerator

from .generation_verbalizer import GenerationVerbalizer
from .soft_verbalizer import SoftVerbalizer
from .prototypical_verbalizer import ProtoVerbalizer

TEMPLATE_CLASS = {
    'manual_template': ManualTemplate,
    'mixed_template': MixedTemplate,
    'ptuning_template': PtuningTemplate,
    'soft_template': SoftTemplate,
    'ptr_template': PTRTemplate,
    'prefix_tuning_template': PrefixTuningTemplate,
}

VERBALIZER_CLASS = {
    'manual_verbalizer': ManualVerbalizer,
    'knowledgeable_verbalizer': KnowledgeableVerbalizer,
    'automatic_verbalizer': AutomaticVerbalizer,
    'ptr_verbalizer': PTRVerbalizer,
    'one2one_verbalizer': One2oneVerbalizer,
    'generation_verbalizer': GenerationVerbalizer,
    'soft_verbalizer': SoftVerbalizer,
    'proto_verbalizer': ProtoVerbalizer,
}

TEMPLATE_GENERATOR_CLASS = {
    't5': T5TemplateGenerator
}

VERBALIZER_GENERATOR_CLASS = {
    'roberta': RobertaVerbalizerGenerator
}


def load_template(config: CfgNode,
                **kwargs,
                ):
    r"""
    Args:
        config: (:obj:`CfgNode`) The global configure file.
        kwargs: kwargs might include:
                plm_model: Optional[PreTrainedModel],
                plm_tokenizer: Optional[PreTrainedTokenizer],
                plm_config: Optional[PreTrainedConfig]

    Returns:
        A template
    """
    if config.template is not None:
        template_class = TEMPLATE_CLASS[config.template]
        template = template_class.from_config(config=config[config.template],
                                     **kwargs)
    return template

def load_verbalizer(config: CfgNode,
                **kwargs,
                ):
    r"""
    Args:
        config: (;obj:`CfgNode`) The global configure file.
        kwargs: kwargs might include:
                plm_model: Optional[PreTrainedModel],
                plm_tokenizer: Optional[PreTrainedTokenizer],
                plm_config: Optional[PreTrainedConfig]

    Returns:
        A template
    """
    if config.verbalizer is not None:
        verbalizer_class = VERBALIZER_CLASS[config.verbalizer]
        verbalizer = verbalizer_class.from_config(config=config[config.verbalizer],
                                     **kwargs)
    return verbalizer

def load_template_generator(config: CfgNode, **kwargs,):
    template_generator = None
    if config.classification.auto_t:
        template_generator_class = TEMPLATE_GENERATOR_CLASS[config.template_generator.plm.model_name]
        template_generator = template_generator_class.from_config(config=config.template_generator, **kwargs)
    return template_generator

def load_verbalizer_generator(config: CfgNode, **kwargs,):
    verbalizer_generator = None
    if config.classification.auto_v:
        verbalizer_generator_class = VERBALIZER_GENERATOR_CLASS[config.plm.model_name]
        verbalizer_generator = verbalizer_generator_class.from_config(config=config.verbalizer_generator, **kwargs)
    return verbalizer_generator
