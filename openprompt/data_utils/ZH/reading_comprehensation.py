from .processor import *

class C3(DataProcessor):
    """
    @article{sun2020investigating,
      title={Investigating prior knowledge for challenging chinese machine reading comprehension},
      author={Sun, Kai and Yu, Dian and Yu, Dong and Cardie, Claire},
      journal={Transactions of the Association for Computational Linguistics},
      volume={8},
      pages={141--155},
      year={2020},
      publisher={MIT Press}
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for example_json in json.load(f):
                example = InputExample(
                    meta = {
                        "text": example_json[0][0],
                        "question": example_json[1][0]["question"],
                        "options": example_json[1][0]["choice"],
                    },
                    tgt_text = example_json[1][0]["choice"].index(example_json[1][0]["answer"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 问题: {question} {options}'
        ]


class CCPM(DataProcessor):
    """
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
                        "text": example_json["translation"],
                        "options": example_json["choices"],
                    },
                    tgt_text = example_json["answer"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '释义：{text} 问题: 这句释义对应的古文是? {options}',
        ]


class SPP(DataProcessor):
    """
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                story = example_json["story"]
                modified = []
                count = 0
                i = 0
                while i < len(story):
                    if story[i: i+6] == '[MASK]':
                        count += 1
                        modified.append(f"[空白{count}]")
                        i = i + 6
                    else:
                        modified.append(story[i])
                        i = i + 1
                        
                example = InputExample(
                    meta = {
                        "story": "".join(modified),
                        "plot": example_json["sentence"],
                        "blank": "[空白]",
                        "options": ",".join(f'[空白{i}]' for i in range(1, count+1))
                    },
                    tgt_text = int(example_json["label"])
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '故事：{story} 情节：{plot} 问题：上述情节应该填充到故事中的哪一个{blank}处？{options}',
        ]


class CMedQA(DataProcessor):
    """
    @article{zhang2017chinese,
    title={Chinese Medical Question Answer Matching Using End-to-End Character-Level Multi-Scale CNNs},
    author={Zhang, Sheng and Zhang, Xin and Wang, Hui and Cheng, Jiajun and Li, Pei and Ding, Zhaoyun},
    journal={Applied Sciences},
    volume={7},
    number={8},
    pages={767},
    year={2017},
    publisher={Multidisciplinary Digital Publishing Institute}
    }

    @ARTICLE{8548603, 
    author={S. Zhang and X. Zhang and H. Wang and L. Guo and S. Liu}, 
    journal={IEEE Access}, 
    title={Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection}, 
    year={2018}, 
    volume={6}, 
    number={}, 
    pages={74061-74071}, 
    keywords={Biomedical imaging;Data mining;Semantics;Medical services;Feature extraction;Knowledge discovery;Medical question answering;interactive attention;deep learning;deep neural networks}, 
    doi={10.1109/ACCESS.2018.2883637}, 
    ISSN={2169-3536}, 
    month={},}
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        questions = {}
        answers = {}
        with open(os.path.join(data_dir, "questions.csv"), encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i==0: continue
                q_id, content = row[:2]
                questions[q_id] = content
        with open(os.path.join(data_dir, "answers.csv"), encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i==0: continue
                a_id, _, content = row
                answers[a_id] = content

        path = os.path.join(data_dir, f"{split}_candidates.txt")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i==0: continue
                q_id, pos_id, neg_id = row
                example = InputExample(
                    meta = {
                        "question": questions[q_id],
                        "options": [answers[pos_id], answers[neg_id]],
                    },
                    tgt_text = 0,
                )
                examples.append(example)
                example = InputExample(
                    meta = {
                        "question": questions[q_id],
                        "options": [answers[neg_id], answers[pos_id]],
                    },
                    tgt_text = 1,
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '问题：{question} {options}'
        ]


class ChiD(DataProcessor):
    """
    @article{DBLP:journals/corr/abs-1906-01265,
    author    = {Chujie Zheng and
                Minlie Huang and
                Aixin Sun},
    title     = {ChID: {A} Large-scale Chinese IDiom Dataset for Cloze Test},
    journal   = {CoRR},
    volume    = {abs/1906.01265},
    year      = {2019},
    url       = {http://arxiv.org/abs/1906.01265},
    eprinttype = {arXiv},
    eprint    = {1906.01265},
    timestamp = {Thu, 14 Oct 2021 09:16:22 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-1906-01265.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}_answer.json")
        answer_map = {}
        with open(path, encoding='utf8') as f:
            answer_map = json.load(f)
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                candidates = example_json["candidates"]
                for content in example_json["content"]:
                    modified = []
                    answers = []
                    i = 0
                    while i < len(content):
                        if content[i: i+6] == '#idiom':
                            modified.append(f"[空白]")
                            j = i+6
                            while j < len(content):
                                if content[j]=='#': break
                                j = j+1
                            j += 1
                            answers.append(candidates[answer_map[content[i:j]]])
                            i = j
                        else:
                            modified.append(content[i])
                            i = i + 1
                            
                    example = InputExample(
                        meta = {
                            "text": "".join(modified),
                            "blank": "[空白]",
                            "candidates": ",".join(candidates),
                        },
                        tgt_text = ",".join(answers)
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文段：{text} 成语：{candidates} 问题：文段的{blank}处应依次填入哪些成语? 回答：',
        ]


class CSL(CLSProcessor):
    """
    中文科技文献数据集(CSL)
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1",
            ],
            labels_mapped = [
                "不是", "是",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                            
                example = InputExample(
                    meta = {
                        "text": example_json["abst"],
                        "keywords": ",".join(example_json["keyword"]),
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label"])
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文章摘要：{text} 关键词：{keywords} 问题：上述关键词和这篇文章关联吗？回答：'
        ]

class CJRC(DataProcessor):
    """
    @InProceedings{10.1007/978-3-030-32381-3_36,
    author="Duan, Xingyi and Wang, Baoxin and Wang, Ziyue and Ma, Wentao and Cui, Yiming and Wu, Dayong and Wang, Shijin and Liu, Ting and Huo, Tianxiang and Hu, Zhen and Wang, Heng and Liu, Zhiyuan",
    editor="Sun, Maosong and Huang, Xuanjing and Ji, Heng and Liu, Zhiyuan and Liu, Yang",
    title="CJRC: A Reliable Human-Annotated Benchmark DataSet for Chinese Judicial Reading Comprehension",
    booktitle="Chinese Computational Linguistics",
    year="2019",
    publisher="Springer International Publishing",
    address="Cham",
    pages="439--451",
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            jsons = json.load(f)
            for example_json in jsons["data"]:
                for paragraph in example_json["paragraphs"]:
                    for qa in paragraph["qas"]:
                        if qa["is_impossible"]=='true': continue
                        example = InputExample(
                            meta = {
                                "casename": paragraph["casename"],
                                "context": paragraph["context"],
                                "question": qa["question"],
                            },
                            tgt_text = qa["answers"][0]["text"],
                        )
                        examples.append(example)
                
        

    def get_templates(self):
        return [
            '法律案例：{casename}:{context} 问题：根据上述法律案例，{question} 回答：'
        ]


class CMRC2018(DataProcessor):
    """
    @inproceedings{cui-emnlp2019-cmrc2018,
        title = "A Span-Extraction Dataset for {C}hinese Machine Reading Comprehension",
        author = "Cui, Yiming  and
        Liu, Ting  and
        Che, Wanxiang  and
        Xiao, Li  and
        Chen, Zhipeng  and
        Ma, Wentao  and
        Wang, Shijin  and
        Hu, Guoping",
        booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
        month = nov,
        year = "2019",
        address = "Hong Kong, China",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D19-1600",
        doi = "10.18653/v1/D19-1600",
        pages = "5886--5891",
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"data/{split}.json")
        
        with open(path, encoding='utf8') as f:
            jsons = json.load(f)
            for example_json in jsons:
                for qa in example_json["qas"]:
                    example = InputExample(
                        meta = {
                            "context": example_json["context_text"],
                            "question": qa["query_text"],
                        },
                        tgt_text = qa["answers"][0],
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题：根据上文，{question} 回答：'
        ]


class CMRC2019(DataProcessor):
    """
    @inproceeding={cui-etal-2020-cmrc2019,
        title={A Sentence Cloze Dataset for Chinese Machine Reading Comprehension},
        author={Cui, Yiming and Liu, Ting and Yang, Ziqing and Chen, Zhipeng and Ma, Wentao and Che, Wanxiang and Wang, Shijin and Hu, Guoping},
        booktitle = 	"Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)",
        year={2020}
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"data/{split}.json")
        
        with open(path, encoding='utf8') as f:
            jsons = json.load(f)["data"]
            for example_json in jsons:
                example = InputExample(
                    meta = {
                        "story": example_json["context"].replace('[BLANK', '[空白'),
                        'blank': '[空白]',
                        "plots": ",".join(example_json["choices"]),
                    },
                    tgt_text = ",".join([example_json["choices"][a] for a in example_json["answers"]])
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '故事：{story} 情节:{plots} 问题：故事中{blank}处依次填入什么情节？回答：'
        ]


class DuReader(DataProcessor):
    """
    @article{DBLP:journals/corr/abs-1711-05073,
    author    = {Wei He and
                Kai Liu and
                Yajuan Lyu and
                Shiqi Zhao and
                Xinyan Xiao and
                Yuan Liu and
                Yizhong Wang and
                Hua Wu and
                Qiaoqiao She and
                Xuan Liu and
                Tian Wu and
                Haifeng Wang},
    title     = {DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world
                Applications},
    journal   = {CoRR},
    volume    = {abs/1711.05073},
    year      = {2017},
    url       = {http://arxiv.org/abs/1711.05073},
    eprinttype = {arXiv},
    eprint    = {1711.05073},
    timestamp = {Wed, 02 Dec 2020 18:07:16 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-1711-05073.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        for which in ["search", "zhidao"]:
            path = os.path.join(data_dir, f"{split}set/{which}.{split}.json")
            
            with open(path, encoding='utf8') as f:
                for line in f:
                    example_json = json.loads(line)
                    example = InputExample(
                        meta = {
                            "context": ";".join([para for doc in example_json["documents"] for para in doc["paragraphs"]]),
                            "question": example_json["question"],
                        },
                        tgt_text = ";".join(example_json["answers"])
                    )
                    examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{context} 问题:根据上文，{question} 回答：',
        ]