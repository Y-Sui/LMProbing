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


## pawsx dataset
#python run_paswx.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#python run_paswx.py --model_name_or_path "xlm-roberta-base" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
## finetune on xnli using xlm model
#python run_xnli.py --model_name_or_path "xlm-roberta-base" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#python run_xnli.py --model_name_or_path "bert-base-multilingual-uncased" --language en --train_language en --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1


#pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh' 'all_languages')
#xnli_Languages=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh' 'all_languages')
#for lang in ${pawsx_Languages[@]}; do
#    # pawsx dataset
#    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#done
#for lang in ${xnli_Languages[@]}; do
## finetune on xnli using xlm model
#    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#done


#python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --language all_languages --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --language all_languages --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1


#pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh' 'all_languages')
#xnli_Languages=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh' 'all_languages')
#for lang in ${pawsx_Languages[@]}; do
#    # pawsx dataset
#    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert_cat" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base_cat" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#done
#for lang in ${xnli_Languages[@]}; do
#    # finetune on xnli using xlm model
#    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base_cat" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#
#    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert_cat" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#done


#python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en-all" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en-all" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en-all" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
#python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --language en --train_language all_languages --do_train --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en-all" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1


pawsx_Languages=('en' 'de' 'es' 'fr' 'ja' 'ko' 'zh' 'all_languages')
xnli_Languages=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh' 'all_languages')
for lang in ${pawsx_Languages[@]}; do
    # pawsx dataset
    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --language $lang  --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
    python run_paswx.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 128 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
done
for lang in ${xnli_Languages[@]}; do
    # finetune on xnli using xlm model
    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
    python run_xnli.py --model_name_or_path "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --language $lang --do_eval --num_train_epochs 2.0 --max_seq_length 256 --overwrite_output_dir  --output_dir "//home/weicheng/data_interns/yuansui/models/baseline" --save_steps 10000 --per_gpu_train_batch_size 32 --learning_rate 5e-5  --warmup_steps 1200 --weight_decay 0.1
done
