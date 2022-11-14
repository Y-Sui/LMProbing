#!/bin/bash

pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh')
#ud_subsets=()

# 1. NER task
for lang in ${pawsx_Languages[@]}; do

  # M-BERT task
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"

  # XLM-R task
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"

done

## 2. UD task (part of speech, and dependency task
#for lang in ${ud_subsets[@]}$; do
#
#  # M-BERT task
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#  python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#
#  # XLM-R task
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#  python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=layer-wise --fc=probing --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=1e-3 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#
#done