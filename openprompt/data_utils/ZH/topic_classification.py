from .processor import *

class THUCNews(CLSProcessor):
    """
    @inproceedings{sun2016THUCNews,
    title={THUCTC: An Efficient Chinese Text Classifier},
    author={Maosong Sun, Jingyang Li, Zhipeng Guo, Yu Zhao, Yabin Zheng, Xiance Si, Zhiyuan Liu},
    year={2016}
    }
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐",
            ],
            labels_mapped = [
                "财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐",
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.jsonl")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "headline": example_json["headline"],
                        "content": example_json["context"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["class"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '新闻标题：{headline} 新闻正文：{content} 问题：上述新闻属于什么什么类别？{options}'
        ]


class TNews(CLSProcessor):
    """
    CLUE
    """
    def __init__(self):
        super().__init__(
            labels_origin = [
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"
            ],
            labels_mapped = [
                "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞", "民生"
            ]
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i==0: continue
                label, text = row
                example = InputExample(
                    meta = {
                        "text": text,
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(label),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '新闻：{text} 问题：上述新闻属于什么什么类别？{options}'
        ]


class IFLYTEK(CLSProcessor):
    """
    CLUE
    """
    def __init__(self):
        labels_des = [
            "打车", "地图导航", "免费WIFI", "租车", "同城服务", "快递物流", "婚庆", "家政", "公共交通", "政务", "社区服务", "薅羊毛", "魔幻", "仙侠", "卡牌", "飞行空战", "射击游戏", "休闲益智", "动作类", "体育竞技", "棋牌中心", "经营养成", "策略", "MOBA", "辅助工具", "约会社交", "即时通讯", "工作社交", "论坛圈子", "婚恋社交", "情侣社交", "社交工具", "生活社交", "微博博客", "新闻", "漫画", "小说", "技术", "教辅", "问答交流", "搞笑", "杂志", "百科", "影视娱乐", "求职", "兼职", "视频", "短视频", "音乐", "直播", "电台", "K歌", "成人", "中小学", "职考", "公务员", "英语", "视频教育", "高等教育", "成人教育", "艺术", "语言(非英语)", "旅游资讯", "综合预定", "民航", "铁路", "酒店", "行程管理", "民宿短租", "出国", "工具", "亲子儿童", "母婴", "驾校", "违章", "汽车咨询", "汽车交易", "日常养车", "行车辅助", "租房", "买房", "装修家居", "电子产品", "问诊挂号", "养生保健", "医疗服务", "减肥瘦身", "美妆美业", "菜谱", "餐饮店", "体育咨讯", "运动健身", "支付", "保险", "股票", "借贷", "理财", "彩票", "记账", "银行", "美颜", "影像剪辑", "摄影修图", "相机", "绘画", "二手", "电商", "团购", "外卖", "电影票务", "社区超市", "购物咨询", "笔记", "办公", "日程管理", "女性", "经营", "收款", "其他",
        ]
        super().__init__(
            labels_origin = labels_des,
            labels_mapped = labels_des,
        )

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.json")
        
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                example = InputExample(
                    meta = {
                        "text": example_json["sentence"],
                        "options": self.labels_mapped,
                    },
                    tgt_text = self.get_label(example_json["label_des"]),
                )
                examples.append(example)
                
        

    def get_templates(self):
        return [
            '文本：{text} 问题：上述文段属于什么什么类别？{options}'
        ]