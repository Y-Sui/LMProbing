#!/bin/bash

# This script is used for fine-tuning each languages' evaluation using the M-BERT and XLM-R models
# Save the sota checkpoints with epochs=15, model_load_path=..

# Supported Languages
xnli_Languages=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh')

# 1. M-BERT model
for lang in ${xnli_Languages[@]}; do

  # Probing task
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3

  # finetune (only consider XNLI & PAWS-X)
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=$lang --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=$lang --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5

done

# 2. XLM-R model
for lang in ${pawsx_Languages[@]}; do

  # Probing task
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3

  # finetune (only consider XNLI & PAWS-X)
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=$lang --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=$lang --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5


done




