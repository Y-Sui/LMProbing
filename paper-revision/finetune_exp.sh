# Finetune mbert and xlm-r on xnli and pawsx dataset


# pawsx dataset
#python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --save_steps 10000 --per_gpu_train_batch_size 32
#python run_paswx.py --model_name_or_path "xlm-roberta-base" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --save_steps 10000 --per_gpu_train_batch_size 32

## finetune on xnli using xlm model
#python run_xnli.py --model_name_or_path "xlm-roberta-base" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir True --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
## finetune on xnli using mbert model
#python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --save_steps 10000 --per_gpu_train_batch_size 32
#
## finetune on pawsx mbert model
#python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir True --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1

## Test raw model without finetuning
#python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --do_eval --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline"
#python run_xnli.py --model_name_or_path "xlm-roberta-base" --language en --do_eval --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline"
#python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --do_eval --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline"
#python run_paswx.py --model_name_or_path "xlm-roberta-base" --language en --do_eval --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline"




# pawsx dataset
python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1

python run_paswx.py --model_name_or_path "xlm-roberta-base" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1

# finetune on xnli using xlm model
python run_xnli.py --model_name_or_path "xlm-roberta-base" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1

python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
