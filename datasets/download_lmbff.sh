#!/bin/sh
rm -rf LMBFF
wget --content-disposition https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar -O LMBFF.tar
tar -xvf LMBFF.tar
rm -f LMBFF.tar