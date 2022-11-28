# Finetune mbert and xlm-r on xnli and pawsx dataset

# xnli dataset
python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --save_steps 500 --per_gpu_train_batch_size 32
python run_xnli.py --model_name_or_path "xlm-roberta-large" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm" --save_steps 500 --per_gpu_train_batch_size 32

# pawsx dataset
python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --save_steps 500 --per_gpu_train_batch_size 32
python run_paswx.py --model_name_or_path "xlm-roberta-large" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm" --save_steps 500 --per_gpu_train_batch_size 32

