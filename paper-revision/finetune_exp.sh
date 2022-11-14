# finetune M-Bert, XLM-R on xnli and pawsx dataset

python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=pawsx --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/pawsx/M-BERT_finetune

python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=pawsx --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/pawsx/XLM-R_finetune

python run.py --model_config=M-BERT --tokenizer_config=M-BERT --corpus=xnli --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/xnli/M-BERT_finetune

python run.py --model_config=XLM-R --tokenizer_config=XLM-R --corpus=xnli --lang=all_languages --fc=finetune --embed_size=large --max_length=128 --batch_size=32 --seed=1234 --epochs=2 --lr=5e-5 --checkpoints=/home/weicheng/data_interns/yuansui/models/xnli/XLM-R_finetune

