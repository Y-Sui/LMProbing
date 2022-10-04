#!/usr/bin/python

python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234
python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "ner" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234
python run.py --task "ner" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234
python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234

python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234
python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 30 --batch_size 1024 --max_length 50 --lr 0.0001 --seed 1234

