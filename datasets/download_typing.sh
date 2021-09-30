#!/bin/sh
DIR="./Typing"
mkdir $DIR
cd $DIR

rm -rf FewNERD
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bcacdddd54c44c5e86b1/?dl=1
tar -zxvf FewNERD.tar.gz
rm -rf FewNERD.tar.gz

cd ..