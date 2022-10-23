#!/usr/bin/python

python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 30 --batch_size 64 --max_length 50 --lr 0.0001 --seed 1234
