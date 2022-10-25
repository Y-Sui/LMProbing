#!/usr/bin/python

# finetune (only consider XNLI & PAWS-X)
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=en --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=en --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001

python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=en --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=en --mode=layer-wise --fc=finetune --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001


# probing task
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=ud --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001

python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=ud --lang=en --mode=layer-wise --fc=probing --embed_size=large --max_length=100 --batch_size=128 --seed=1234 --epochs=3 --lr=0.0001
