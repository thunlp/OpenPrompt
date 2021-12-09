from .processor import *

class Mandarinograd(CLSProcessor):
    """
    @inproceedings{bernard2020mandarinograd,
    title={Mandarinograd: A Chinese Collection of Winograd Schemas},
    author={Bernard, Timoth{\'e}e and Han, Ting},
    booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
    pages={21--26},
    year={2020}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "不可以", "可以", 
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for example_json in json.load(f).values():
                example = InputExample(
                    meta = {
                        "before": example_json["text"],
                        "after": example_json["hypothesis"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["class_id"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{before} 问题：上述文本进行指代消解后，可以理解为 "{after}" 吗？{options}',
        ]


class CLUEWSC2020(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "false", "true",
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
                        "that": example_json["target"]["span1_text"],
                        "it": example_json["target"]["span2_text"],
                        "text": example_json["text"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 问题：在上文中，“{it}“ 指代 "{that}" 吗？{options}',
        ]