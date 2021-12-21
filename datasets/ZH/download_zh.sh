DIR="./ZH"
mkdir $DIR
cd $DIR

rm -rf coreference
wget --content-disposition https://cloud.tsinghua.edu.cn/f/c0c2b5eac1ad49e49e9b/?dl=1
unzip coreference
rm coreference.zip

rm -rf entity_typing
wget --content-disposition https://cloud.tsinghua.edu.cn/f/f0a6d6a40fdb4f309b4a/?dl=1
unzip entity_typing
rm entity_typing.zip

rm -rf nli
wget --content-disposition https://cloud.tsinghua.edu.cn/f/3b2828bf2d164e2eb326/?dl=1
unzip nli
rm nli.zip

rm -rf paraphrase
wget --content-disposition https://cloud.tsinghua.edu.cn/f/c1e22934a1ca4e58a47b/?dl=1
unzip paraphrase
rm paraphrase.zip

rm -rf relation
wget --content-disposition https://cloud.tsinghua.edu.cn/f/8c6dbc98af8b417e8690/?dl=1
unzip relation
rm relation.zip

rm -rf sentiment
wget --content-disposition https://cloud.tsinghua.edu.cn/f/1fe676fb68174862b1c8/?dl=1
unzip sentiment
rm sentiment.zip

rm -rf topic_classification
wget --content-disposition https://cloud.tsinghua.edu.cn/f/c891f5b65bc442c59427/?dl=1
unzip topic_classification
rm topic_classification

rm -rf reading_comprehension
wget --content-disposition https://cloud.tsinghua.edu.cn/f/3fc2ff5b6b4d40178c6a/?dl=1
unzip reading_comprehension
rm reading_comprehension

rm -rf __MACOSX

cd ..