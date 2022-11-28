#!/bin/bash

pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh')
ud_subsets=('ar_pud' 'bg_btb' 'zh_gsdsimp' 'nl_alpino' 'en_gum' 'fr_pud' 'el_gdt' 'de_pub' 'hi_pud' 'es_pud' 'ru_pud' 'tr_pub' 'vi_vtb' 'th_pud' 'ur_udtb' 'ja_pud' 'ko_pud')

# 1. NER task
for lang in ${pawsx_Languages[@]}; do

  # M-BERT task
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"

  # XLM-R task
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=xnli --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"

done

## 2. UD task (part of speech, and dependency task
#for lang in ${ud_subsets[@]}$; do
#
#  # M-BERT task
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#  python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
#
#  # XLM-R task
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm"
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=PER --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=LOC --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#  python run_probing.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=wikiann --tag_class=ORG --lang=$lang --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm"
#
#done

# test
python run_probing.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=ud --tag_class=ORG --lang=ar_pud --mode=run --fc=probing --embed_size=large --max_length=32 --batch_size=64 --seed=1234 --epochs=8 --lr=1e-1 --src=pawsx --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --task=POS
