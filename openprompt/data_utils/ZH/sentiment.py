from .processor import *

class ChnSentiCorp(CLSProcessor):
    """
    @inproceedings{st2008ChnSentiCorp,
    title={An empirical study of sentiment analysis for chinese documents},
    booktitle={Expert Systems with Applications},
    pages={2612--2619}
    author={Songbo Tan, Jin Zhang},
    year={2008}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "消极", "积极",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "context": example_json["text_a"],
                        "options": self.labels_mapped,
                    },
                    label = self.get_label(example_json["label"]),
                )
                examples.append(example)
        return examples
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题:上述文本所表达的情感为？{options}',
        ]


class ECISA(CLSProcessor):
    """
    @article{徐琳宏2008情感词汇本体的构造,
    title={情感词汇本体的构造},
    author={徐琳宏 and 林鸿飞 and 潘宇 and 任惠 and 陈建美},
    journal={情报学报},
    volume={27},
    number={2},
    pages={180--185},
    year={2008}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1", "2",
            ],
            labels_mapped = [
                "不含情感", "褒义", "贬义",
            ]
        )

    def get_examples(self, data_dir, split):
        if split == 'dev': return []
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for example_json in json.load(f):
                sents = example_json["Sentence"]
                if isinstance(sents, dict): sents = [sents]
                for i, sent in enumerate(sents):
                    if "label" in sent:
                        example = InputExample(
                            meta = {
                                "context_before": "".join([
                                    s["text"] for s in sents[:i]
                                ]),
                                "text": sent["text"],
                                "context_after": "".join([
                                    s["text"] for s in sents[i+1:]
                                ]),
                                "options": self.labels_mapped,
                            },
                            tgt_text = self.get_label(sent["label"]),
                        )
                        examples.append(example)
                
        
        
    def get_templates(self):
        return [
            '文本：{context_before} {text} {context_after} 问题：上述文本中，"{text}"所表达的情感为？{options}',
        ]


class JD_FULL(CLSProcessor):
    """
    @article{zx2017JDFull,
    author    = {Xiang Zhang and Yann LeCun},
    title     = {Which Encoding is the Best for Text Classification in Chinese, English, Japanese and Korean?},
    journal   = {CoRR},
    volume    = {abs/1708.02657},
    year      = {2017},
    url       = {http://arxiv.org/abs/1708.02657},
    archivePrefix = {arXiv},
    eprint    = {1708.02657}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "1", "2", "3", "4", "5",
            ],
            labels_mapped = [
                "1", "2", "3", "4", "5",
                # "很差 ", "差", "中", "好", "很好",
            ]
        )

    def get_examples(self, data_dir, split):
        if split == 'dev': return []
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "title": example_json["title"],
                        "review": example_json["review"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        
        
    def get_templates(self):
        return [
            '评价：{title} {review} 问题：据此分析，这段评价给出的评分为？{options}'
        ]


class SimplifyWeibo4Moods(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1", "2", "3"
            ],
            labels_mapped = [
                "喜悦", "愤怒", "厌恶", "低落"
            ]
        )

    def get_examples(self, data_dir, split):
        if split!='train': return []
        path = os.path.join(data_dir, f"{split}.csv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                label, review = row[0], ",".join(row[1:])
                example = InputExample(
                    meta = {
                        "context": review,
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(label),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上述文本所表达的情感为？{options}',
        ]


class PositiveNegative(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "消极", "积极",
            ]
        )

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.csv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                label, review = row[0], ",".join(row[1:])
                example = InputExample(
                    meta = {
                        "context": review,
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(label),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上述文本所表达的情感为？{options}',
        ]


class RatingMovie(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "1", "2", "3", "4", "5",
            ],
            labels_mapped = [
                "1", "2", "3", "4", "5",
            ]
        )

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.csv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                rating, text = row[2], row[4] 
                example = InputExample(
                    meta = {
                        "text": text,
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(rating),
                )
                examples.append(example)
                
        
        
    def get_templates(self):
        return [
            '评价：{text} 问题：据此估计，这段对电影的评价对应的评分为？{options}'
        ]


class RatingShopping(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "1", "2", "3", "4", "5",
            ],
            labels_mapped = [
                "1", "2", "3", "4", "5",
            ]
        )

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.csv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                rating, text = row[2], row[4]+","+row[5]
                if rating not in self.labels_origin: continue # illegal
                example = InputExample(
                    meta = {
                        "text": text,
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(rating),
                )
                examples.append(example)
                
        
        
    def get_templates(self):
        return [
            '评价：{text} 问题：这段评价对商品的评分为？{options}'
        ]


class RatingDianping(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "1.0", "2.0", "3.0", "4.0", "5.0",
            ],
            labels_mapped = [
                "1", "2", "3", "4", "5",
            ]
        )

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.csv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                rating_env, rating_flavor, rating_service, text = row[3], row[4], row[5], row[7]
                if rating_env=="": continue # illegal data
                example = InputExample(
                    meta = {
                        "text": text,
                        "question": "环境",
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(rating_env)
                )
                examples.append(example)
                example = InputExample(
                    meta = {
                        "text": text,
                        "question": "特色",
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(rating_flavor)
                )
                examples.append(example)
                example = InputExample(
                    meta = {
                        "text": text,
                        "question": "设施",
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(rating_service)
                )
                examples.append(example)
        
        
    def get_templates(self):
        return [
            '评价:{text} 问题：据此估计，这段评价对{question}的评分为？{options}'
        ]