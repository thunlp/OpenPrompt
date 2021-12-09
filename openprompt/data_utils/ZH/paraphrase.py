from .processor import *

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
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text_1": example_json["text_1"],
                        "text_2": example_json["text_2"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文段一：{text_1} 文段二：{text_2} 问题：上述两个文段有什么关系？{options}',
        ]


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
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text_1": example_json["sentence1"],
                        "text_2": example_json["sentence2"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]) if "label" in example_json else None,
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文段一：{text_1} 文段二：{text_2} 问题：上述两个文段表达的意思是否一致？{options}',
        ]
