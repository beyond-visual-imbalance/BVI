#!/bin/bash

python train.py  /WIT/en_es/en-es-10k \
		--arch transformer_iwslt_de_en \
		--share-decoder-input-output-embed \
		--clip-norm 0 \
		--optimizer adam \
		--reset-optimizer \
		--lr 0.001 \
		--source-lang en \
		--target-lang es \
		--max-tokens 4096 \
		--no-progress-bar \
		--log-interval 100 \
		--weight-decay 0.3 \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.3 \
		--lr-scheduler inverse_sqrt \
		--max-update 8000 \
		--warmup-updates 2000 \
		--warmup-init-lr 1e-07 \
		--update-freq 4 \
		--adam-betas 0.9,0.98 \
		--keep-last-epochs 20 \
		--dropout 0.3 \
		--tensorboard-logdir results/en-de-bpe/bl_log1 \
		--log-format simple \
		--save-dir results \
		--eval-bleu \
		--eval-bleu-remove-bpe \
		--patience 15 \
		--fp16 \




