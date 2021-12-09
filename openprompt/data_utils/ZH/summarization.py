from .processor import *

class CEPSUM(DataProcessor):
    """
    @inproceedings{yuan-etal-2020-faithfulness,
    title = "On the Faithfulness for {E}-commerce Product Summarization",
    author = "Yuan, Peng  and Li, Haoran  and Xu, Song  and Wu, Youzheng  and He, Xiaodong  and Zhou, Bowen",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.502",
    doi = "10.18653/v1/2020.coling-main.502",
    pages = "5712--5717",
    abstract = "In this work, we present a model to generate e-commerce product summaries. The consistency between the generated summary and the product attributes is an essential criterion for the ecommerce product summarization task. To enhance the consistency, first, we encode the product attribute table to guide the process of summary generation. Second, we identify the attribute words from the vocabulary, and we constrain these attribute words can be presented in the summaries only through copying from the source, i.e., the attribute words not in the source cannot be generated. We construct a Chinese e-commerce product summarization dataset, and the experimental results on this dataset demonstrate that our models significantly improve the faithfulness.",
    }
    """
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                kvs = example_json["table"].split('\t')
                kvs = [(kvs[i], kvs[i+1]) for i in range(0, len(kvs), 2)]
                example = InputExample(
                    meta = {
                        "properties": ",".join([f"{k}:{v}" for k, v in kvs]),
                        "description": example_json["source"],
                    },
                    tgt_text = example_json["targets"][0], # TODO on dev and test, there's more than one answer available
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '产品属性：{properties} 产品描述：{description} 目标：根据产品的属性和描述，为该产品写一个摘要。 摘要：'
        ]


class LCSTS(DataProcessor):
    """
    @inproceedings{hu2015lcsts,
    title={LCSTS: A Large Scale Chinese Short Text Summarization Dataset},
    author={Hu, Baotian and Chen, Qingcai and Zhu, Fangze},
    booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
    pages={1967--1972},
    year={2015}
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
                        "text": example_json["text"],
                    },
                    tgt_text = example_json["summary"],
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 目标：为上述文本写一个摘要。摘要：'
        ]
