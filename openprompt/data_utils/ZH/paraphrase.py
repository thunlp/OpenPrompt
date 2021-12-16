from .processor import *

class LCQMC(CLSProcessor):
    """
    @inproceedings{liu-etal-2018-lcqmc,
    title = "{LCQMC}:A Large-scale {C}hinese Question Matching Corpus",
    author = "Liu, Xin  and
      Chen, Qingcai  and
      Deng, Chong  and
      Zeng, Huajun  and
      Chen, Jing  and
      Li, Dongfang  and
      Tang, Buzhou",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/C18-1166",
    pages = "1952--1962",
    abstract = "The lack of large-scale question matching corpora greatly limits the development of matching methods in question answering (QA) system, especially for non-English languages. To ameliorate this situation, in this paper, we introduce a large-scale Chinese question matching corpus (named LCQMC), which is released to the public1. LCQMC is more general than paraphrase corpus as it focuses on intent matching rather than paraphrase. How to collect a large number of question pairs in variant linguistic forms, which may present the same intent, is the key point for such corpus construction. In this paper, we first use a search engine to collect large-scale question pairs related to high-frequency words from various domains, then filter irrelevant pairs by the Wasserstein distance, and finally recruit three annotators to manually check the left pairs. After this process, a question matching corpus that contains 260,068 question pairs is constructed. In order to verify the LCQMC corpus, we split it into three parts, i.e., a training set containing 238,766 question pairs, a development set with 8,802 question pairs, and a test set with 12,500 question pairs, and test several well-known sentence matching methods on it. The experimental results not only demonstrate the good quality of LCQMC but also provide solid baseline performance for further researches on this corpus.",
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "无关", "相似",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i==0: continue
                text_a, text_b, label = row
                example = InputExample(
                    text_a = text_a,
                    text_b = text_b,
                    meta = {
                        "options": self.labels_mapped,
                    },
                    label = self.get_label(label),
                )
                examples.append(example)
        return examples

class ChineseSTS(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0.0", "1.0", "2.0", "3.0", "4.0", "5.0",
            ],
            labels_mapped = [
                "完全没关系", "基本没关系", "没什么关系", "有点关系", "基本一样", "完全一样",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text_1": example_json["text_1"],
                        "text_2": example_json["text_2"],
                        "options": self.labels_mapped,
                    },
                    label = self.get_label(example_json["label"]),
                )
                examples.append(example)
        return examples
                
    # def get_templates(self):
    #     return [
    #         '文段一：{text_1} 文段二：{text_2} 问题：上述两个文段有什么关系？{options}',
    #     ]


class AFQMC(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "否", "是",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text_1": example_json["sentence1"],
                        "text_2": example_json["sentence2"],
                        "options": self.labels_mapped,
                    },
                    label = self.get_label(example_json["label"]) if "label" in example_json else None,
                )
                examples.append(example)
        return examples
                
    # def get_templates(self):
    #     return [
    #         '文段一：{text_1} 文段二：{text_2} 问题：上述两个文段表达的意思是否一致？{options}',
    #     ]
