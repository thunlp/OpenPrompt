#!/bin/sh
rm -rf SNLI
wget --content-disposition https://cloud.tsinghua.edu.cn/f/c72b5ee3f992490d8846/?dl=1
tar -zxvf SNLI.tar.gz
rm -rf SNLI.tar.gz