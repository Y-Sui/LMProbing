import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4" # set the cuda card 2,3,4,5
CUDA_DEVICES = 4

import argparse
import logging
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from datasets import load_metric
from dataset.config import DataConfig
from model import Bert_4_Classification_Head_Wise, Bert_4_Classification_Layer_Wise
from dataloader import get_files_path, get_sequence_classification

import torch.multiprocessing as mp
import torch.distributed as dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ner", type=str, help="Please specify the task name {NER or Chunk}")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="Path to save the pretrained model")
    parser.add_argument("--embed_size", default=256, type=int)
    parser.add_argument("--label_size", default=2, type=int, help="classification task: the number of the label classes")
    parser.add_argument("--corpus", default="//home/weicheng/data_interns/yuan/", type=str)
    # Options parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name_or_path", )
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name_or_path", )
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--profile", action="store_true", help="whether to generate the heatmap")
    parser.add_argument("--mode", choices=["layer-wise", "head-wise"], type=str, help="choose training mode", default="layer-wise")
    parser.add_argument('--nprocs', default=CUDA_DEVICES, type=int, metavar='N',
                        help='number of processes, one for each GPU. (torch.cuda.device_count())')
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    return args


def get_double_repetitive_item(items):
    if not isinstance(items, list):
        return None
    temp = []
    double_items = []
    for item in items:
        if item not in temp:
            if items.count(item) == 2:
                double_items.append(item)
            else:
                pass
        else:
            temp.append(item)
    return double_items


def train(eval_model, train_loader, eval_loader, label_list, file_path, mode, label, epochs,
          device, profile, task, lr):
    logger = logging.getLogger("Eval-probing-training")
    logger.info("Train() started!")
    sample_config = DataConfig()
    if mode == "layer-wise":
        logging_path = os.path.join(sample_config.logging_path, task, "bert_classification_layer_wise")
    else:
        logging_path = os.path.join(sample_config.logging_path, task, "bert_classification_head_wise")
    loop_size = len(eval_model.hidden_states) if mode == "layer-wise" else eval_model.num_heads * len(eval_model.hidden_states)
    final_score = []
    for i in range(loop_size):  # i refers to head or layer
        if mode == "layer-wise":
            model = Bert_4_Classification_Layer_Wise(num_labels=len(label_list))
        else:
            model = Bert_4_Classification_Head_Wise(num_labels=len(label_list)).to(device)
        model = torch.nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # remove special token
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),  # only update the fc parameters (classifier)
            lr=lr,
        )
        optimizer.zero_grad()  # make sure each layer's optimizer set to zero grad
        for epoch in range(1, epochs + 1):
            for idx, train_batches in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = train_batches["input_ids"].to(device)
                attention_mask = train_batches["attention_mask"].to(device)
                labels = train_batches["labels"].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs[i]
                # adapt to the nn.crossentropy, inputs = [batch_size, nb_classes, *additional_dims]; target in the shape [batch_size, *additional_dims]
                preds = logits.permute(0, 2, 1).to(
                    device)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                if idx % 5 == 0:
                    logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                    wandb.log({
                        "epoch": epoch,
                        "batch": idx,
                        f"train/loss_{label}": loss.data,
                        f"{mode}-th": i
                    })
        torch.save(model.state_dict(), os.path.join(sample_config.checkpoints, f"{mode}_idx_{i}.pt"))
        logger.info(f"{mode} {i} on {file_path} has been trained..")
        logger.info(f"start to evaluate the model..")
        eval_results = eval(i, model, eval_loader, label_list, file_path, mode, device)
        final_score.append(eval_results)

    # Save the evaluation
    logger.info(f"save the evaluation score to {logging_path}/{mode}_{file_path}.csv")
    profile_logging = []
    with open(os.path.join(logging_path, f"{label}_{mode}_{file_path}.csv"), "w") as file:
        acc, recall, f1, prec = [], [], [], []
        for i in range(len(final_score)):
            logger.info(f"{label} Performance of the {i}th is:")
            logger.info(f"{label} precision: {final_score[i]['overall_precision']}")
            logger.info(f"{label} Recall: {final_score[i]['overall_recall']}")
            logger.info(f"{label} F1, {final_score[i]['overall_f1']}")
            logger.info(f"{label} Accuracy, {final_score[i]['overall_accuracy']}")

            wandb.log({
                f"valid/{label}/acc": final_score[i]['overall_accuracy'],
                f"valid/{label}/prec": final_score[i]['overall_precision'],
                f"valid/{label}/f1": final_score[i]['overall_f1'],
                f"valid/{label}/recall": final_score[i]['overall_recall'],
                f"{mode}-th": i
            })

            acc.append(final_score[i]['overall_accuracy'])
            recall.append(final_score[i]['overall_recall'])
            f1.append(final_score[i]['overall_f1'])
            prec.append(final_score[i]['overall_precision'])
            # generate the heatmap according to the F1 score
            profile_logging.append(final_score[i]['overall_f1'])

        metric_frame = pd.DataFrame({f"{mode}": [i for i in range(len(final_score))],
                                     "Accuracy": acc, "Precision": prec, "Recall": recall, "F1": f1})
        metric_frame.to_csv(file, index=False, sep=",")
    logger.info(f"{label} metrics evaluation has been saved!")

    # Save the profile figure of the output
    if profile:
        logger.info(f"Start to generate the {label} {mode}/{file_path} profile png..")
        if mode == "head-wise":
            final_score = np.reshape(profile_logging, (model.num_heads, len(model.hidden_states)))
            final_score = pd.DataFrame(final_score, index=[f"head_{i}" for i in range(model.num_heads)],
                                       columns=[f"layer_{j}" for j in range(len(model.hidden_states))])
            sns_fig = sns.heatmap(final_score, vmin=0, vmax=1, cmap="YlGnBu")
        elif mode == "layer-wise":
            x_ = [f"{i}" for i in range(len(model.hidden_states))]
            sns_fig = sns.barplot(x=x_, y=profile_logging, color="#42b7bd")
            patch_h = [patch.get_height() for patch in sns_fig.patches]
            idx_tallest = np.argmax(patch_h)
            sns_fig.patches[idx_tallest].set_facecolor('#a834a8')
        plt.savefig(os.path.join(sample_config.output_path, task, f"{label}_{mode}_{file_path}_map.png"))
        plt.clf()

    return final_score


def eval(index, model, eval_loader, label_list, file_path, mode, device):
    logger = logging.getLogger("Eval-probing-evaluation")
    logger.info("Eval() started!")
    with torch.no_grad():
        model.to(device)
        metric = load_metric("seqeval")
        for example_batched in tqdm(eval_loader):
            input_ids = example_batched["input_ids"].to(device)
            attention_mask = example_batched["attention_mask"].to(device)
            labels = example_batched["labels"].int().to(device)  # use int()
            outputs = model(input_ids, attention_mask)
            logits = outputs[index]  # CLS
            preds = torch.argmax(logits, dim=2).int().to(device)  # use int()
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds, labels)
            ]
            metric.add_batch(predictions=true_predictions, references=true_labels)
        results = metric.compute()

        logger.info(f"{mode} {index} on {file_path} has been evaluated..")
        return results


def main(args):
    logger = logging.getLogger(f"Eval-probing(@{os.getpid()})")
    sample_config = DataConfig()
    pos_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    neg_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "neg_samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    for i in range(len(pos_sample_path["train"])):

        file_name = pos_sample_path['train'][i].split('/')[-1].replace('_train', '')

        # Wandb init
        project = 'Eval Probing'
        entity = 'yuansui'
        group = 'Eval-probing-for-layer-wise-and-head-wise'
        display_name = f"task[{args.task}/{file_name.replace('.json', '')}]-mode[{args.mode}]"
        wandb.init(reinit=True, project=project, entity=entity,
                   name=display_name, group=group, tags=["train & eval"])
        wandb.config["args"] = vars(args)
        wandb.config["dataset"] = f"{file_name.replace('.json', '')}"

        # set the data loader
        logger.info(f"Constructing the dataloader for {args.task}/{file_name.replace('.json', '')}")
        probing_train_dataloader, probing_eval_dataloader, probing_label_list = get_sequence_classification(i, "pos")

        logger.info(
            f"Constructing the negative dataloader for {args.task}/{neg_sample_path['train'][i].split('/')[-1].replace('_train', '')}")
        neg_probing_train_dataloader, neg_probing_eval_dataloader, neg_probing_label_list = get_sequence_classification(i, "neg")

        pos_final_score, neg_final_score = [], []

        # load the model
        if args.mode == "layer-wise":
            model_layer_wise = Bert_4_Classification_Head_Wise(num_labels=len(probing_label_list))
            logger.info(f"{args.mode} exp on {args.task} for positive samples starts")
            pos_final_score = train(model_layer_wise, probing_train_dataloader, probing_eval_dataloader,
                                    probing_label_list, file_name, mode=args.mode, label="pos", epochs=args.epochs,
                                    device=args.device, profile=args.profile, task=args.task, lr=args.lr)
            logger.info(f"{args.mode} exp on {args.task} for negative samples starts")
            neg_final_score = train(model_layer_wise, neg_probing_train_dataloader, neg_probing_eval_dataloader,
                                    neg_probing_label_list, neg_sample_path['train'][i].split('/')[-1].replace('_train', ''),
                                    mode=args.mode, label="neg", epochs=args.epochs, device=args.device, profile=args.profile,
                                    task=args.task, lr=args.lr)
        elif args.mode == "head-wise":
            model_head_wise = Bert_4_Classification_Head_Wise(num_labels=len(probing_label_list))
            logger.info(f"{args.mode} exp on {args.task} for positive samples starts")
            pos_final_score = train(model_head_wise, probing_train_dataloader, probing_eval_dataloader,
                                    probing_label_list, file_name, mode=args.mode, label="pos", epochs=args.epochs,
                                    device=args.device, profile=args.profile, task=args.task, lr=args.lr)
            logger.info(f"{args.mode} exp on {args.task} for negative samples starts")
            neg_final_score = train(model_head_wise, neg_probing_train_dataloader, neg_probing_eval_dataloader,
                                    neg_probing_label_list, neg_sample_path['train'][i].split('/')[-1].replace('_train', ''),
                                    mode=args.mode, label="neg", epochs=args.epochs, device=args.device, profile=args.profile,
                                    task=args.task, lr=args.lr)
        else:
            logger.warning("Unsupported mode")

        for i in range(len(pos_final_score)):
            logger.info(f"pos-neg Performance of the {i}th is:")
            logger.info(
                f"pos-neg precision: {pos_final_score[i]['overall_precision'] - neg_final_score[i]['overall_precision']}")
            logger.info(
                f"pos-neg Recall: {pos_final_score[i]['overall_recall'] - neg_final_score[i]['overall_recall']}")
            logger.info(f"pos-neg F1, {pos_final_score[i]['overall_f1'] - neg_final_score[i]['overall_f1']}")
            logger.info(
                f"pos-neg Accuracy, {pos_final_score[i]['overall_accuracy'] - neg_final_score[i]['overall_accuracy']}")

            wandb.log({
                f"valid/gap/acc": pos_final_score[i]['overall_accuracy'] - neg_final_score[i]['overall_accuracy'],
                f"valid/gap/prec": pos_final_score[i]['overall_precision'] - neg_final_score[i]['overall_precision'],
                f"valid/gap/f1": pos_final_score[i]['overall_f1'] - neg_final_score[i]['overall_f1'],
                f"valid/gap/recall": pos_final_score[i]['overall_recall'] - neg_final_score[i]['overall_recall'],
                f"{args.mode}-th": i
            })

        logger.info("finish")
        wandb.log({"finish": True})
        wandb.finish()


if __name__ == "__main__":
    args = get_arguments()
    logger = logging.getLogger("Eval Probing")
    logger.info(f"Args: {args}")
    main(args)