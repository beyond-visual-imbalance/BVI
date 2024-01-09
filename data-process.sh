#!/bin/bash

python3 preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref /WIT/en-de/bpe10000/train \
  --validpref /WIT/en-de/bpe10000/valid \
  --testpref /WIT/en-de/bpe10000/test \
  --nwordssrc 17200 \
  --nwordstgt 9800 \
  --workers 12 \
  --destdir /WIT/en-de \
#  --srcdict data-bin/en-de/test2016/dict.en.txt \
#  --tgtdict data-bin/en-de/test2016/dict.de.txt