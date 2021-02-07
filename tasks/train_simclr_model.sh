#!/bin/bash
python training/run_simclr.py --train_mode=pretrain \
  --train_batch_size=128 --train_epochs=120 \
  --learning_rate=0.01 --weight_decay=1e-4 --temperature=1.0 \
 --image_size=128 --eval_split=test --resnet_depth=18 \
  --use_blur=True \
  --model_dir=./tmp