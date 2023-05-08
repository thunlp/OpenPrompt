#!/bin/sh
DIR="./datasets/TextClassification"
mkdir $DIR
cd $DIR

# rm -rf mnli
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/33182c22cb594e88b49b/?dl=1
# tar -zxvf mnli.tar.gz
# rm -rf mnli.tar.gz

# rm -rf agnews
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/0fb6af2a1e6647b79098/?dl=1
# tar -zxvf agnews.tar.gz
# rm -rf agnews.tar.gz

# rm -rf dbpedia
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/362d3cdaa63b4692bafb/?dl=1
# tar -zxvf dbpedia.tar.gz
# rm -rf dbpedia.tar.gz

# rm -rf imdb
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
# tar -zxvf imdb.tar.gz
# rm -rf imdb.tar.gz

# rm -rf SST-2
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/bccfdb243eca404f8bf3/?dl=1
# tar -zxvf SST-2.tar.gz
# rm -rf SST-2.tar.gz

# rm -rf amazon
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/e00a4c44aaf844cdb6c9/?dl=1
# tar -zxvf amazon.tar.gz
# mv datasets/amazon/ amazon
# rm  -rf ./datasets
# rm -rf amazon.tar.gz

# rm -rf yahoo_answers_topics
# wget --content-disposition https://cloud.tsinghua.edu.cn/f/79257038afaa4730a03f/?dl=1
# tar -zxvf yahoo_answers_topics.tar.gz
# rm -rf yahoo_answers_topics.tar.gz

rm -rf dwmw17
FILEID="1FW_qQX8aubnuFy--y8cY8HW26CFixFei"
FILENAME="dwmw17.zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
unzip dwmw17.zip
rm dwmw17.zip

cd ..
