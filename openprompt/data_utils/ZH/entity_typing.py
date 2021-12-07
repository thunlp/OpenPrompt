from .processor import *

class CMeEE_NER(CLSProcessor):
    """
    @inproceedings{hongying2020building,
    title={Building a Pediatric Medical Corpus: Word Segmentation and Named Entity Annotation},
    author={Hongying, Zan and Wenxin, Li and Kunli, Zhang and Yajuan, Ye and Baobao, Chang and Zhifang, Sui},
    booktitle={Workshop on Chinese Lexical Semantics},
    pages={652--664},
    year={2020},
    organization={Springer}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep",
            ],
            labels_mapped = [
                "疾病", "临床表现", "药物", "医疗设备", "医疗程序", "身体", "医学检验项目", "微生物类", "科室",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        examples = []
        with open(path, encoding='utf8') as f:
            for example_json in json.load(f):
                for span in example_json["entities"]:

                    example = InputExample(
                        meta = {
                            "context": example_json["text"],
                            "entity": span["entity"],
                            "options": self.labels_mapped,
                        },
                        label = self.get_label(span["type"])
                    )
                    examples.append(example)
        return examples
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上文中，实体“{entity}”是什么类型的? {options}',
        ]


class Resume_NER(CLSProcessor):
    """
    @article{zhang2018chinese,  
    title={Chinese NER Using Lattice LSTM},  
    author={Yue Zhang and Jie Yang},  
    booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2018}  
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "NAME",  "CONT", "LOC", "RACE", "PRO", "EDU", "ORG", "TITLE",
            ],
            labels_mapped = [
                "人名", "国籍", "籍贯", "种族", "专业", "学位", "机构", "职称",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonline")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                for span in example_json["span_list"]:
                    example = InputExample(
                        meta = {
                            "context": "".join(example_json["tokens"]),
                            "entity":  "".join(example_json["tokens"][span["start"]: span["end"]+1]),
                            "options": self.labels_mapped,
                        },
                        tgt_text = self.get_label(span["type"]),
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上文中，实体“{entity}”是什么类型的? {options}',
        ]


class Weibo_NER(CLSProcessor):
    """
    @inproceedings{peng2015ner,
    title={Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings},
    author={Peng, Nanyun and Dredze, Mark},
    booktitle={Processings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    pages={548–-554},
    year={2015}, File={https://www.aclweb.org/anthology/D15-1064/}, }

    @inproceedings{peng2016improving,
    title={Improving named entity recognition for Chinese social media with word segmentation representation learning},
    author={Peng, Nanyun and Dredze, Mark},
    booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)},
    volume={2},
    pages={149--155},
    year={2016}, File={https://www.aclweb.org/anthology/P16-2025/}, }

    @inproceedings{he-sun-2017-f,
    title = "{F}-Score Driven Max Margin Neural Network for Named Entity Recognition in {C}hinese Social Media",
    author = "He, Hangfeng  and
    Sun, Xu",
    booktitle = "Proceedings of the 15th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 2, Short Papers",
    month = apr,
    year = "2017",
    address = "Valencia, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/E17-2113",
    pages = "713--718",
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [ # TODO 区分 NAM 特指 NOM 泛指
                "PER.NAM",  "PER.NOM", "LOC.NAM", "LOC.NOM", "ORG.NAM", "ORG.NOM", "GPE.NAM", "GPE.NOM", 
            ],
            labels_mapped = [
                "人", "泛指人", "地点", "泛指地点", "机构", "泛指机构", "地理政治实体", "泛指地理政治实体", 
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonline")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                for span in example_json["span_list"]:
                    example = InputExample(
                        meta = {
                            "context": "".join(example_json["tokens"]),
                            "entity": "".join(example_json["tokens"][span["start"]: span["end"]+1]),
                            "options": self.labels_mapped,
                        },
                        tgt_text = self.get_label(span["type"]),
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上文中，实体“{entity}”是什么类型的? {options}',
        ]


class DH_MSRA(CLSProcessor):
    """
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "PER", "LOC", "ORG"
            ],
            labels_mapped = [
                "人", "地点", "机构"
            ]
        )

    def get_examples(self, data_dir, split):
        if split != 'train': raise ValueError
        path = os.path.join(data_dir, f"{split}.txt")
        
        with open(path, encoding='utf8') as f:
            xs, ys = [], []
            for line in f:
                l = line.split()
                if len(l)==0:
                    i = 0
                    while i < len(xs):
                        if ys[i][0] == 'B':
                            j = i + 1
                            while j < len(xs):
                                if ys[j][0] == 'O': break
                                j = j + 1

                            example = InputExample(
                                meta = {
                                    "context": "".join(xs),
                                    "entity": "".join(xs[i:j]),
                                    "options": self.labels_mapped,
                                },
                                tgt_text = self.get_label(ys[i][2:]),
                            )
                            examples.append(example)
                            
                            i = j
                        else:
                            i = i + 1

                    xs, ys = [], []
                else:
                    xs.append(l[0])
                    ys.append(l[1])

                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：上文中，实体“{entity}”是什么类型的? {options}',
        ]
