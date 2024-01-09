#!/bin/bash

python3 generate.py     /Fashion-MMT \
				--path results/mmtimg1/model.pt \
				--source-lang en --target-lang zh \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results \
				--remove-bpe \
#				--fp16 \