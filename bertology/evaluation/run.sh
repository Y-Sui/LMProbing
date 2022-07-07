#!/bin/bash
python layer_wise_run_bert.py --model_name_or_path bert-base-uncased --output_dir ../../output/bert_classification --batch_size 4 --epochs 5 --max_length 50 --seed 1234