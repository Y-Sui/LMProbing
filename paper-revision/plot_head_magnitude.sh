##########################################
# en
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --normalize
# all languages
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --normalize
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --normalize
# raw
python head_magnitude.py --checkpoints "bert-base-multilingual-uncased" --normalize
python head_magnitude.py --checkpoints "xlm-roberta-base" --normalize

# difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --normalize --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --normalize --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --normalize --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --normalize --plot_difference

# #############################################
# unnormalize

# en
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en"
# all languages
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base"
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base"
# raw
python head_magnitude.py --checkpoints "bert-base-multilingual-uncased"
python head_magnitude.py --checkpoints "xlm-roberta-base"

python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --plot_difference
python head_magnitude.py --checkpoints "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --plot_difference


#################################
# raw difference
python head_magnitude.py --checkpoints "bert-base-multilingual-uncased" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert" --plot_difference
python head_magnitude.py --checkpoints "bert-base-multilingual-uncased" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert" --plot_difference
python head_magnitude.py --checkpoints "xlm-roberta-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base" --plot_difference
python head_magnitude.py --checkpoints "xlm-roberta-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base" --plot_difference

python head_magnitude.py --checkpoints "bert-base-multilingual-uncased" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-mbert-en" --plot_difference
python head_magnitude.py --checkpoints "bert-base-multilingual-uncased" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert-en" --plot_difference
python head_magnitude.py --checkpoints "xlm-roberta-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-xnli-xlm-base-en" --plot_difference
python head_magnitude.py --checkpoints "xlm-roberta-base" --checkpoints2 "//home/weicheng/data_interns/yuansui/models/finetune-pawsx-xlm-base-en" --plot_difference
