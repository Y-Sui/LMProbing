#!/usr/bin/python

# Conduct layer-wise experiments over bert-base-uncased
python run.py --task "ner" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --mode=layer-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234

# Conduct head-wise experiments over bert-base-uncased
python run.py --task "ner" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "chunk" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "dependency-fixed" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234
python run.py --task "pos-fixed" --model_name_or_path "bert-base-uncased" --mode=head-wise --epochs 8 --batch_size 64 --max_length 64 --lr 1e-4 --seed 1234





