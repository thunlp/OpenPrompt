.. _data_processors:

Data Processors
==================================


Basic Processor
---------------------------------

Abstract class that provides methods for loading train/dev/test/unlabeled examples for a given task.

.. autoclass:: openprompt.data_utils.data_processor.DataProcessor
   :members:


Text Classification Processor
---------------------------------


AgnewsProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.text_classification_dataset.AgnewsProcessor
   :members:


DBpediaProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.text_classification_dataset.DBpediaProcessor
   :members:

ImdbProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.text_classification_dataset.ImdbProcessor
   :members:

SST2Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.text_classification_dataset.SST2Processor
   :members:

..
   AmazonProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.text_classification_dataset.AmazonProcessor
      :members:

..
   LAMA Processor
   ---------------------------------

   LAMAProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.lama_dataset.LAMAProcessor


..
   SuperGlue Processor
   ---------------------------------


   FewGLUEDataProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.FewGLUEDataProcessor
      :members:


   RteProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.RteProcessor
      :members:
      


   CbProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.CbProcessor
      :members:


   WicProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.WicProcessor
      :members:
      


   WscProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.WscProcessor
      :members:
      

   BoolQProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.BoolQProcessor
      :members:
      

   CopaProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.CopaProcessor
      :members:

   MultiRcProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.MultiRcProcessor
      :members:


   RecordProcessor
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: openprompt.data_utils.superglue_dataset.RecordProcessor
      :members:



Entity Typing Processor
---------------------------------

FewNERDProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.typing_dataset.FewNERDProcessor


Relation Classification Processor
---------------------------------

TACREDProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.relation_classification_dataset.TACREDProcessor

TACREVProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.relation_classification_dataset.TACREVProcessor


ReTACREDProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.relation_classification_dataset.ReTACREDProcessor

SemEvalProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.relation_classification_dataset.SemEvalProcessor



Language Inference Processor
---------------------------------

SNLIProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.nli_dataset.SNLIProcessor


Conditional Generation Processor
---------------------------------

WebNLGProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openprompt.data_utils.conditional_generation_dataset.WebNLGProcessor
   :members:

