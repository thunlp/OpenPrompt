#!/bin/sh
DIR="./TextClassification"
mkdir $DIR
cd $DIR

rm -rf mnli
wget --content-disposition https://cloud.tsinghua.edu.cn/f/ac761f94ab194483b3ba/?dl=1
tar -zxvf mnli.tar.gz
rm -rf mnli.tar.gz

rm -rf agnews
wget --content-disposition https://cloud.tsinghua.edu.cn/f/27c27ad244404e368ee7/?dl=1
tar -zxvf agnews.tar.gz
rm -rf agnews.tar.gz

rm -rf dbpedia
wget --content-disposition https://cloud.tsinghua.edu.cn/f/1db4c5ddd2474c4ea79d/?dl=1
tar -zxvf dbpedia.tar.gz
rm -rf dbpedia.tar.gz

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/b4dda7c843bf4647ad82/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

cd ..