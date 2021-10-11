from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import *
from openprompt.utils.logging import logger

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric: Optional[str] = "micro-f1",
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score

def generation_metric(hypos,
                      refs, 
                      metric: Optional[str] = "sentence_bleu"):
    r"""Some basic metric function for generation. However, many generation tasks
    has their own evaluation bash scripts.

    Args:
        hypos (:obj:`str`) : the generated sentence.
        refs (:obj:`list(str)`) : the referenced (ground-truth) sentence.
        metric (:obj:`str`, `optional`) : the type of metric option

    Returns:
        score (float): evaluate score
    """
    if metric == "sentence_bleu":
        # a simple criterion to visualize the performance, not rigorous.
        import nltk
        try:
            nltk_path = str(nltk.data.find("tokenizers/punkt"))
            logger.info(f"using nltk from: {nltk_path}")
        except LookupError:
            nltk.download('punkt')

        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method4 # a function for smooth
        scores = []
        
        for ref, hypo in zip(refs, hypos):
            tokenized_rs = []
            ref = ref.split("\n")
            for r in ref:
                tokenized_rs.append(word_tokenize(r))
            hypo = word_tokenize(hypo)
            try:
                sc = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
            except ValueError: # TODO ZeroDivisionError
                logger.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
                sc = 0.0
            scores.append(sc)
        score = sum(scores)/len(scores)
        return score
    else:
        raise ValueError("'{}' is not a valid metric type.".format(metric))