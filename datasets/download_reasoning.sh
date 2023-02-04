#!/bin/sh
# CSQA
DIR="./Reasoning"
mkdir $DIR
cd $DIR

DIR="./csqa"
mkdir $DIR
cd $DIR

wget --content-disposition https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
wget --content-disposition https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget --content-disposition https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl

cd ..


cd .. 