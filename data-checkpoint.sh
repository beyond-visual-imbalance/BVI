#!/bin/bash


python3 scripts/average_checkpoints.py \
			--inputs results/mmtimg1 \
			--num-epoch-checkpoints 5 \
			--output results/mmtimg1/model.pt \

