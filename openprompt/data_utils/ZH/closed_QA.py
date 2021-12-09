from .processor import *

class ZhiDao(DataProcessor):
    """
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.csv")
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i==0: continue
                title, question, reply, is_best = row
                if not is_best: continue
                example = InputExample(
                    meta = {
                        "title": title,
                        "question": question,
                    },
                    tgt_text = reply,
                )
                examples.append(example)

    def get_templates(self):
        return [
            '问题：{title} {question} 回答：',
        ]