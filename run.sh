#!/usr/bin/python

python run.py --task "ner" --model_name_or_path "bert-base-uncased" --epochs 8 --batch_size 32 --max_length 50 --lr 0.0001 --seed 1234 --profile

python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --epochs 8 --batch_size 32 --max_length 50 --lr 0.0001 --seed 1234 --profile

python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --epochs 8 --batch_size 32 --max_length 50 --lr 0.0001 --seed 1234 --profile

python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --epochs 8 --batch_size 32 --max_length 50 --lr 0.0001 --seed 1234 --profile

