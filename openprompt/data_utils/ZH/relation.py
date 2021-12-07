from .processor import *

class FinRE(CLSProcessor):
    """
    @inproceedings{li-etal-2019-FinRE,
    title = "{C}hinese Relation Extraction with Multi-Grained Information and External Linguistic Knowledge",
    author = "Li, Ziran  and Ding, Ning  and Liu, Zhiyuan  and Zheng, Haitao  and Shen, Ying",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1430",
    doi = "10.18653/v1/P19-1430",
    pages = "4377--4386",
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            ],
            labels_mapped = [
                "无关系", "注资", "拥有", "纠纷", "同一个", "增持", "重组", "买资", "签约", "持股", "交易", "入股", "转让", "成立", "分析", "合作", "帮助", "发行", "商讨", "合并", "竞争", "订单", "减持", "合资", "收购", "借壳", "欠款", "被发行", "被转让", "被成立", "被注资", "被持股", "被拥有", "被收购", "被帮助", "被借壳", "被买资", "被欠款", "被增持", "拟收购", "被减持", "被分析", "被入股", "被拟收购",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text": example_json["text"],
                        "head": example_json["head"]["mention"],
                        "tail": example_json["tail"]["mention"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 问题:上述文本中，“{head}”和“{tail}”的关系为？{options}'
        ]


class Chinese_Literature_NER_RE(CLSProcessor):
    """
    @article{xu2017discourse,
    title={A discourse-level named entity recognition and relation extraction dataset for chinese literature text},
    author={Xu, Jingjing and Wen, Ji and Sun, Xu and Su, Qi},
    journal={arXiv preprint arXiv:1711.07010},
    year={2017}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            ],
            labels_mapped = [
                "位于", "局部整体", "家人", "普遍特殊", "社会", "所有者", "使用", "制作", "邻近",
            ] #Located, Part-Whole, Family, General-Special, Social, Ownership, Use, Create, Near
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                for relation in example_json["relations"]:
                    example = InputExample(
                        meta = {
                            "text": example_json["text"],
                            "head": example_json["entities"][relation["head"]][0]["mention"],
                            "tail": example_json["entities"][relation["tail"]][0]["mention"],
                            "options": self.labels_mapped,
                        },
                        tgt_text = self.get_label(relation["label"]),
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 问题:上述文本中，“{head}”和“{tail}”的关系为？{options}'
        ]
