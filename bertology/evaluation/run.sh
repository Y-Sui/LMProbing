#!/bin/bash
python layer_wise_run_bert_SST2.py --model_name_or_path bert-base-uncased --dataset SST2 --output_dir ../../output/bert_classification --batch_size 16 --epochs 5 --max_length 50 --seed 1234
python head_wise_run_bert_SST2.py --model_name_or_path bert-base-uncased --dataset SST2 --output_dir ../../output/bert_classification --batch_size 16 --epochs 5 --max_length 50 --seed 1234

set -e
zip the ./output and upload to the OSS
zip -q -r output.zip ./output
oss cp output.zip oss://backup/