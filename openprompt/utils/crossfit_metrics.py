# Used in tutorial 4.1, where all tasks are completed in generation fashion
# This scripts is used for evaluation of the generation text and convert it into the metric of the original task.
# directly copied from https://github.com/INK-USC/CrossFit/blob/ce47dfa9478d2d19e7176888ee1f39413b3bd91c/dataloader/metrics.py#L241
# Thanks to the authors of CrossFit for the interesting paper (https://arxiv.org/abs/2104.08835) and awesome  project.

import numpy as np
import string
import re
from collections import Counter
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from rouge import Rouge

METRICS = {
    'acronym_identification': 'EM',
    'ade_corpus_v2-classification': 'Classification-F1',
    'ade_corpus_v2-dosage': 'EM',
    'ade_corpus_v2-effect': 'EM',
    'adversarialqa': 'QA-F1',
    'aeslc': 'Rouge-L',
    'ag_news': 'Classification-F1',
    'ai2_arc': 'ACC',
    'amazon_polarity': 'Classification-F1',
    'anli': 'Classification-F1',
    'app_reviews': 'Pearson-Correlation',
    'aqua_rat': 'ACC',
    'art': 'ACC',
    'aslg_pc12': 'EM',
    'biomrc': 'QA-F1',
    'blimp-anaphor_gender_agreement': 'ACC',
    'blimp-anaphor_number_agreement': 'ACC',
    'blimp-determiner_noun_agreement_with_adj_irregular_1': 'ACC',
    'blimp-ellipsis_n_bar_1': 'ACC',
    'blimp-ellipsis_n_bar_2': 'ACC',
    'blimp-existential_there_quantifiers_1': 'ACC',
    'blimp-irregular_past_participle_adjectives': 'ACC',
    'blimp-sentential_negation_npi_licensor_present': 'ACC',
    'blimp-sentential_negation_npi_scope': 'ACC',
    'blimp-wh_questions_object_gap': 'ACC',
    'boolq': 'ACC',
    'break-QDMR': 'EM',
    'break-QDMR-high-level': 'EM',
    'circa': 'Classification-F1',
    'climate_fever': 'Classification-F1',
    'codah': 'Classification-F1',
    'common_gen': 'Rouge-L',
    'commonsense_qa': 'ACC',
    'cos_e': 'Rouge-L',
    'cosmos_qa': 'ACC',
    'crawl_domain': 'EM',
    'crows_pairs': 'ACC',
    'dbpedia_14': 'Classification-F1',
    'definite_pronoun_resolution': 'ACC',
    'discovery': 'Classification-F1',
    'dream': 'ACC',
    'duorc': 'QA-F1',
    'e2e_nlg_cleaned': 'Rouge-L',
    'eli5-askh': 'Rouge-L',
    'eli5-asks': 'Rouge-L', # dev
    'eli5-eli5': 'Rouge-L',
    'emo': 'Classification-F1',
    'emotion': 'Classification-F1',
    'empathetic_dialogues': 'Rouge-L',
    'ethos-directed_vs_generalized': 'Classification-F1',
    'ethos-disability': 'Classification-F1',
    'ethos-gender': 'Classification-F1',
    'ethos-national_origin': 'Classification-F1',
    'ethos-race': 'Classification-F1',
    'ethos-religion': 'Classification-F1',
    'ethos-sexual_orientation': 'Classification-F1',
    'financial_phrasebank': 'Classification-F1',
    'freebase_qa': 'EM',
    'gigaword': 'Rouge-L',
    'glue-cola': 'Matthew-Correlation',
    'glue-mnli': 'ACC',
    'glue-mrpc': 'ACC',
    'glue-qnli': 'ACC',
    'glue-qqp': 'ACC',
    'glue-rte': 'ACC',
    'glue-sst2': 'ACC',
    'glue-wnli': 'ACC',
    'google_wellformed_query': 'ACC',
    'hate_speech18': 'Classification-F1',
    'hate_speech_offensive': 'Classification-F1',
    'hatexplain': 'Classification-F1',
    'health_fact': 'Classification-F1',
    'hellaswag': 'ACC',
    'hotpot_qa': 'QA-F1',
    'imdb': 'Classification-F1',
    'jeopardy': 'EM',
    'kilt_ay2': 'EM',
    'kilt_fever': 'ACC',
    'kilt_hotpotqa': 'EM',
    'kilt_nq': 'EM',
    'kilt_trex': 'EM',
    'kilt_wow': 'Rouge-L',
    'kilt_zsre': 'EM',
    'lama-conceptnet': 'EM',
    'lama-google_re': 'EM',
    'lama-squad': 'EM',
    'lama-trex': 'EM',
    'liar': 'Classification-F1',
    'limit': 'EM',
    'math_qa': 'ACC',
    'mc_taco': 'ACC',
    'medical_questions_pairs': 'ACC',
    'mocha': 'Pearson-Correlation',
    'multi_news': 'Rouge-L',
    'numer_sense': 'EM',
    'onestop_english': 'Classification-F1',
    'openbookqa': 'ACC',
    'paws': 'Classification-F1',
    'piqa': 'ACC',
    'poem_sentiment': 'Classification-F1',
    'proto_qa': 'EM', # here
    'qa_srl': 'EM',
    'qasc': 'ACC',
    'quail': 'ACC',
    'quarel': 'ACC',
    'quartz-no_knowledge': 'ACC',
    'quartz-with_knowledge': 'ACC',
    'quoref': 'QA-F1',
    'race-high': 'ACC',
    'race-middle': 'ACC',
    'reddit_tifu-title': 'Rouge-L',
    'reddit_tifu-tldr': 'Rouge-L',
    'ropes': 'QA-F1',
    'rotten_tomatoes': 'Classification-F1',
    'samsum': 'Rouge-L',
    'scicite': 'Classification-F1',
    'sciq': 'ACC',
    'scitail': 'Classification-F1',
    'search_qa': 'EM',
    'sick': 'Classification-F1',
    'sms_spam': 'Classification-F1',
    'social_i_qa': 'ACC',
    'spider': 'EM',
    'squad-with_context': 'QA-F1',
    'squad-no_context': 'EM',
    'superglue-cb': 'ACC',
    'superglue-copa': 'ACC',
    'superglue-multirc': 'EM',
    'superglue-record': 'QA-F1',
    'superglue-rte': 'ACC',
    'superglue-wic': 'ACC',
    'superglue-wsc': 'ACC',
    'swag': 'ACC',
    'tab_fact': 'Classification-F1',
    'trec': 'Classification-F1',
    'trec-finegrained': 'Classification-F1',
    'tweet_eval-emoji': 'Classification-F1',
    'tweet_eval-emotion': 'Classification-F1',
    'tweet_eval-hate': 'Classification-F1',
    'tweet_eval-irony': 'Classification-F1',
    'tweet_eval-offensive': 'Classification-F1',
    'tweet_eval-sentiment': 'Classification-F1',
    'tweet_eval-stance_abortion': 'Classification-F1',
    'tweet_eval-stance_atheism': 'Classification-F1',
    'tweet_eval-stance_climate': 'Classification-F1',
    'tweet_eval-stance_feminist': 'Classification-F1',
    'tweet_eval-stance_hillary': 'Classification-F1',
    'tweet_qa': 'QA-F1',
    'web_questions': 'EM',
    'wiki_auto': 'Classification-F1',
    'wiki_bio': 'Rouge-L',
    'wiki_qa': 'Classification-F1',
    'wiki_split': 'Rouge-L',
    'wikisql': 'EM',
    'wino_grande': 'ACC',
    'wiqa': 'ACC',
    'xsum': 'Rouge-L',
    'yahoo_answers_topics': 'Classification-F1',
    'yelp_polarity': 'Classification-F1',
    'yelp_review_full': 'Pearson-Correlation'
}

def evaluate(predictions, data, metric, **kwargs):
    def cast_to_float(predictions):
        new_predictions = []
        for prediction in predictions:
            try:
                new_predictions.append(float(prediction.strip()))
            except:
                new_predictions.append(float('NaN'))
        assert len(new_predictions) == len(predictions)
        return new_predictions

    assert len(predictions) == len(data)

    if metric == "EM":
        ems = []
        for (prediction, dp) in zip(predictions, data):
            ems.append(get_exact_match_over_list(prediction, dp))
        return np.mean(ems)
    elif metric == "ACC":
        accs = []
        for (prediction, dp) in zip(predictions, data):
            accs.append(get_accruacy_over_list(prediction, dp, **kwargs))
        return np.mean(accs)
    elif metric == "QA-F1": # haven't be tested
        f1s = []
        for (prediction, dp) in zip(predictions, data):
            f1s.append(get_f1_over_list(prediction, dp))
        return np.mean(f1s)
    elif metric == "Classification-F1":
        return f1_score([dp for dp in data], predictions, average="macro")
    elif metric == "Matthew-Correlation": # haven't be tested
        return get_matthews_corr(data, predictions)
    elif metric == "Pearson-Correlation": # haven't be tested
        predictions = cast_to_float(predictions)
        return pearsonr([float(dp[0]) for dp in data], predictions)[0]
    elif metric == "Rouge-L": # haven't be tested
        rouges = []
        for (prediction, dp) in zip(predictions, data):
            rouges.append(get_rouge_over_list(prediction, dp))
        return np.mean(rouges)

def get_matthews_corr(data, predictions):
    # only cola is using this...?
    new_predictions = []
    for prediction in predictions:
        if prediction.strip() == "acceptable":
            new_predictions.append(1.0)
        else:
            new_predictions.append(0.0)
    new_gold = []
    for dp in data:
        if dp[0] == "acceptable":
            new_gold.append(1.0)
        else:
            new_gold.append(0.0)
    return matthews_corrcoef(new_gold, new_predictions)

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def accuracy(prediction, ground_truth, **kwargs):
    if isinstance(prediction, str) and isinstance(ground_truth, str):
        if kwargs.get("only_compare_prefix", False):
            prediction = prediction[:len(ground_truth)]
        return prediction.lower() == ground_truth.lower()
    else:
        return prediction == ground_truth

def get_rouge_over_list(prediction, groundtruth):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if len(remove_punc(prediction)) == 0:
        return 0.0 # during early stages, it might generate nothing?
    # print(prediction)
    rouge = Rouge()
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]

def get_accruacy_over_list(prediction, groundtruth, **kwargs):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([accuracy(prediction, gt, **kwargs) for gt in groundtruth])
    return accuracy(prediction, groundtruth)

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)

def get_exact_match_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match_over_list(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))