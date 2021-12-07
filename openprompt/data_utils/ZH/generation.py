from .processor import *

class E_reviews(DataProcessor):
    """
    @inproceedings{shao-etal-2019-long,
    title = "Long and Diverse Text Generation with Planning-based Hierarchical Variational Model",
    author = "Shao, Zhihong  and Huang, Minlie  and Wen, Jiangtao  and Xu, Wenfei  and Zhu, Xiaoyan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1321",
    doi = "10.18653/v1/D19-1321",
    pages = "3257--3268",
    abstract = "Existing neural methods for data-to-text generation are still struggling to produce long and diverse texts: they are insufficient to model input data dynamically during generation, to capture inter-sentence coherence, or to generate diversified expressions. To address these issues, we propose a Planning-based Hierarchical Variational Model (PHVM). Our model first plans a sequence of groups (each group is a subset of input items to be covered by a sentence) and then realizes each sentence conditioned on the planning result and the previously generated context, thereby decomposing long text generation into dependent sentence generation sub-tasks. To capture expression diversity, we devise a hierarchical latent structure where a global planning latent variable models the diversity of reasonable planning and a sequence of local latent variables controls sentence realization. Experiments show that our model outperforms state-of-the-art baselines in long and diverse text generation.",
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "key_values": ",".join([f"{k}:{v}" for (k, v) in example_json["feature"] ])
                    },
                    tgt_text = example_json["desc"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            "关键词：{key_values} 目标：根据上述关键词信息，生成一段广告文案. 文案：",
        ]


class OutGen(DataProcessor):
    """
    title, outline -> story
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "title": example_json["title"],
                        "outline": ",".join(example_json["outline"]),
                    },
                    tgt_text = example_json["story"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '标题:{title} 大纲:{outline} 目标:根据上述标题和大纲, 写一个故事。故事：',
        ]


class PC(DataProcessor):
    """
    plot completion
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                story = example_json["story"].split("<MASK>")
                example = InputExample(
                    meta = {
                        "text": story[0] + "[空白]" + story[1],
                        "blank": "[空白]",
                    },
                    tgt_text = example_json["plot"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 目标：根据上文，填充{blank}处缺失的情节。情节：',
        ]


class STORAL1(DataProcessor):
    """
    story -> moral
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "story": example_json["story"],
                    },
                    tgt_text = example_json["moral"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            "故事：{story} 目标：根据故事，提炼一句哲理句。哲理句：",
        ]


class STORAL2(DataProcessor):
    """
    beginning, outline, moral -> story
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "beginning": example_json["beginning"],
                        "outline": ",".join(example_json["outline"]),
                        "moral": example_json["moral"],
                    },
                    tgt_text = example_json["beginning"]+example_json["story"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            "开头：{beginning} 大纲：{outline} 哲理：{moral} 目标：根据上述内容，写出一个故事。故事:",
        ]