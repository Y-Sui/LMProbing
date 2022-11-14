python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --save_steps 500 --per_gpu_train_batch_size 32

python run_xnli.py --model_name_or_path "xlm-roberta-large" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm" --save_steps 500 --per_gpu_train_batch_size 32

python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --save_steps 500 --per_gpu_train_batch_size 32

python run_paswx.py --model_name_or_path "xlm-roberta-large" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm" --save_steps 500 --per_gpu_train_batch_size 32

## finetune M-Bert, XLM-R on xnli and pawsx dataset
#
#python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/pawsx/M-BERT_finetune
#
#python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/pawsx/XLM-R_finetune
#
#python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/xnli/M-BERT_finetune
#
#python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/xnli/XLM-R_finetune
#
