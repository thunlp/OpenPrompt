#!/bin/sh
DIR="./RelationClassification"
mkdir $DIR
cd $DIR

rm -rf SemEval
wget --content-disposition https://cloud.tsinghua.edu.cn/f/7e960bef774c4a04bf8e/?dl=1
tar -zxvf SemEval.tar.gz
rm -rf SemEval.tar.gz

cd ..