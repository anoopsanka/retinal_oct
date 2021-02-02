#!/bin/bash
python training/run_simclr.py --train_mode=pretrain \
  --train_batch_size=128 --train_epochs=1 \
  --learning_rate=0.01 --weight_decay=1e-4 --temperature=0.5 \
 --image_size=128 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=./tmp