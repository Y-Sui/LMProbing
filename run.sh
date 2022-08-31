#!/usr/bin/python

# python run.py --task "ner" --model_name_or_path "bert-base-uncased" --epochs 15 --batch_size 8 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --epochs 3 --batch_size 8 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --epochs 3 --batch_size 8 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --epochs 3 --batch_size 8 --max_length 50 --lr 0.0001 --seed 1234

