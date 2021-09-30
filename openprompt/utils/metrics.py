
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import *

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric_type: Optional[str] = "micro-f1",
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        type (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric_type == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric_type == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric_type == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric_type == "precision":
        score = precision_score(labels, preds)
    elif metric_type == "recall":
        score = recall_score(labels, preds)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric_type))
    return score

def generation_metric(hypos, refs, metric_type: Optional[str] = "bleu"):
    r"""Some basic metric function for generation. However, many generation tasks
    has their own evaluation bash scripts.

    Args:
        hypos (:obj:`str`) : the generated sentence.
        refs (:obj:`str`) : the referenced (ground-truth) sentence.
        metric_type (:obj:`str`, `optional`) : the type of metric option

    Returns:
        score (float): evaluate score
    """
    # a simple criterion to visualize the performance, not rigorous.
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4 # a function for smooth
    scores = []
    
    for ref, hypo in zip(refs, hypos):
        ref = word_tokenize(ref)
        hypo = word_tokenize(hypo)
        scores.append(sentence_bleu([ref], hypo, smoothing_function=smoothie))
    score = sum(scores)/len(scores)
    return score