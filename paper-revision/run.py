import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # set the cuda card 2,3,4,5
CUDA_DEVICES = 4

import argparse
import logging
import torch
from dataloader import DEFALT_DATASETS, DEFALT_LANGUAGES, EvaluationProbing, DEFALT_NUM_LABELS, TAG_DICT_WIKIANN
from model import DEFAULT_MODEL_NAMES, DEFAULT_EMBED_SIZE, MBertLayerWise, MBertHeadWise, XLMRHeadWise, XLMRLayerWise
from trainer import EvalTrainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="M-BERT", choices=DEFAULT_MODEL_NAMES.keys(), type=str,
                        help="Name of the pretrained model")
    parser.add_argument("--tokenizer_config", default="bert-base-multilingual-cased", choices=DEFAULT_MODEL_NAMES.keys(), type=str,
                        help="Name of the pretrained tokenizer")
    parser.add_argument("--corpus", default="pawsx", choices=DEFALT_DATASETS.keys(), type=str)
    parser.add_argument("--lang", default="en", choices=DEFALT_LANGUAGES.values(), type=str)
    parser.add_argument("--embed_size", default=256, choices=DEFAULT_EMBED_SIZE, type=int)
    parser.add_argument("--classifier_num", default=2, choices=[1,2,3,4,5], type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")

    # Options parameters
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--profile", action="store_true", help="whether to generate the heatmap")
    parser.add_argument("--mode", choices=["layer-wise", "head-wise"], type=str, help="choose training mode", default="layer-wise")
    parser.add_argument('--nprocs', default=CUDA_DEVICES, type=int, metavar='N',
                        help='number of processes, one for each GPU. (torch.cuda.device_count())')
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.num_labels = DEFALT_NUM_LABELS[args.corpus]
    return args

def main(args):
    logger = logging.getLogger("Run()")
    dataloader = EvaluationProbing(args).get_dataloader()
    EvalTrainer(args, dataloader, TAG_DICT_WIKIANN).train(args)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
