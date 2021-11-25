#!/bin/sh
DIR="./TextClassification"
mkdir $DIR
cd $DIR

rm -rf mnli
wget --content-disposition https://cloud.tsinghua.edu.cn/f/33182c22cb594e88b49b/?dl=1
tar -zxvf mnli.tar.gz
rm -rf mnli.tar.gz

rm -rf agnews
wget --content-disposition https://cloud.tsinghua.edu.cn/f/0fb6af2a1e6647b79098/?dl=1
tar -zxvf agnews.tar.gz
rm -rf agnews.tar.gz

rm -rf dbpedia
wget --content-disposition https://cloud.tsinghua.edu.cn/f/362d3cdaa63b4692bafb/?dl=1
tar -zxvf dbpedia.tar.gz
rm -rf dbpedia.tar.gz

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

rm -rf SST-2
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bccfdb243eca404f8bf3/?dl=1
tar -zxvf SST-2.tar.gz
rm -rf SST-2.tar.gz

cd ..
