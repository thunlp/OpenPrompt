from .processor import *

class CMNLI(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "entailment", "contradiction", "neutral",
            ],
            labels_mapped = [
                "蕴含", "矛盾", "无关",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        examples = []
        with open(path, encoding='utf8') as f:
            for line_i, line in enumerate(f):
                # print(line_i)
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "premise": example_json["sentence1"],
                        "hypothesis": example_json["sentence2"],
                        "options": self.labels_mapped,
                    },
                    label = self.get_label(example_json["label"]),
                )
                examples.append(example)
        return examples
                
        

    def get_templates(self):
        return [
            '前提：{premise} 假设: {hypothesis} 问题：前提和假设是什么关系? {options}',
        ]

class CSNLI(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "entailment", "contradiction", "neutral",
            ],
            labels_mapped = [
                "蕴含", "矛盾", "无关",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "premise": example_json["sentence1"],
                        "hypothesis": example_json["sentence2"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '前提：{premise} 假设: {hypothesis} 问题：前提和假设是什么关系? {options}',
        ]

class OCNLI(CLSProcessor):
    """
    @inproceedings{ocnli,
	title={OCNLI: Original Chinese Natural Language Inference},
	author={Hai Hu and Kyle Richardson and Liang Xu and Lu Li and Sandra Kuebler and Larry Moss},
	booktitle={Findings of EMNLP},
	year={2020},
	url={https://arxiv.org/abs/2010.05444}
    }
    """

    def __init__(self):
        super().__init__(
            labels_origin = [
                "entailment", "contradiction", "neutral",
            ],
            labels_mapped = [
                "蕴含", "矛盾", "无关",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                if "label" not in example_json or example_json["label"] not in self.labels_origin: continue
                example = InputExample(
                    meta = {
                        "premise": example_json["sentence1"],
                        "hypothesis": example_json["sentence2"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '前提：{premise} 假设: {hypothesis} 问题：前提和假设是什么关系? {options}',
        ]