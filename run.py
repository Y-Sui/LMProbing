import os
# set the cuda card 5
# os.environ["CUDA_VISIBLE_DEVICES"]= "4"


import argparse
import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric
from matplotlib import pyplot as plt
from tqdm import tqdm
from model import Bert_4_Classification_Head_Wise, Bert_4_Classification_Layer_Wise
from dataloader import *
import seaborn as sns
import pandas as pd

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="ner", type=str, help="Please specify the task name {NER or Chunk}")
# parser.add_argument("--dataset", default="SST2", type=str, help="The dataset name, the options can be sst, SST2, etc")
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="Path to save the pretrained model")
parser.add_argument("--embed_size", default=256, type=int)
parser.add_argument("--label_size", default=2, type=int, help="classification task: the number of the label classes")
# Options parameters
parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name_or_path", )
parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name_or_path", )
parser.add_argument("--cache_dir", default=None, type=str, help="Where do you want to store the pre-trained models downloaded from s3", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--profile", action="store_true", help="whether to generate the heatmap")
args = parser.parse_args()

# Setup devices (No distributed training here)
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

# set the save path
layer_wise_path = "../../weicheng/data_interns/yuan/eval-probing/bert_classification_layer_wise/" + args.task + "/"
head_wise_path = "../../weicheng/data_interns/yuan/eval-probing/bert_classification_head_wise/" + args.task + "/"

# layer_wise_path = "output/bert_classification_layer_wise" + args.task + "/"
# head_wise_path = "output/bert_classification_head_wise" + args.task + "/"

if not os.path.exists(layer_wise_path):
    os.mkdir(layer_wise_path)
if not os.path.exists(head_wise_path):
    os.mkdir(head_wise_path)

def get_files_path(filePath):
    """
    return the file list
    """
    raw_files = os.listdir(filePath)
    file_paths = []
    for i in range(len(raw_files)):
        if raw_files[i].find("train") == -1 and raw_files[i].find("eval") == -1:
            file_paths.append(raw_files[i].replace(".csv", ""))
        # else:
        #     raw_files.pop(i)  # remove the elements
    return file_paths


def train(model, train_loader, eval_loader, label_list, file_path, mode="layer-wise", epochs=args.epochs, device=args.device, profile=args.profile):
    model.to(device)
    output_path = layer_wise_path if mode == "layer-wise" else head_wise_path
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # remove specical token
    optimizer = torch.optim.Adam(
        filter(lambda p:p.requires_grad, model.parameters()), # only update the fc parameters (classifier)
        lr=args.lr,
    )
    loop_size = len(model.hidden_states) if mode == "layer-wise" else model.num_heads * len(model.hidden_states)
    final_score = []
    for i in range(loop_size): # i refers to head or layer
        optimizer.zero_grad() # make sure each layer's optimizer set to zero grad
        for epoch in range(1, epochs + 1):
            for idx, example_batched in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = example_batched["input_ids"].to(device)
                attention_mask = example_batched["attention_mask"].to(device)
                labels = example_batched["labels"].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs[i]
                preds = logits.permute(0,2,1).to(device) # adapt to the nn.crossentropy, inputs = [batch_size, nb_classes, *additional_dims]; target in the shape [batch_size, *additional_dims]
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                if idx % 30 == 0:
                    print(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
        # output_path = layer_wise_path if mode == "layer-wise" else head_wise_path
        # torch.save(model.state_dict(), output_path + f"bert_classification_{i}.pt")
        print(f"{mode} {i} on {file_path} has been trained..")
        print(f"start to evaluate the model.. ")
        eval_results = eval(i, model, eval_loader, label_list, file_path, mode, device)
        final_score.append(eval_results)

    # Save the evaluation
    with open(output_path + f"{mode}_{file_path}.txt", "w") as file:
        profile_logging = []
        for i in range(len(final_score)):
            file.write(f"Performance of the {i}th is: "
                       f"precision, {final_score[i]['overall_precision']}, "
                       f"Recall, {final_score[i]['overall_recall']}, "
                       f"F1, {final_score[i]['overall_f1']}, "
                       f"Accuracy, {final_score[i]['overall_accuracy']}" + "\n")
            # generate the heatmap according to the F1 score
            profile_logging.append(final_score[i]['overall_f1'])

    # Save the profile figure of the output
    if profile:
        if mode == "head-wise":
            final_score = np.reshape(profile_logging, (model.num_heads, len(model.hidden_states)))
            final_score = pd.DataFrame(final_score, index=[f"head_{i}" for i in range(model.num_heads)],
                                       columns=[f"layer_{j}" for j in range(len(model.hidden_states))])
            sns_fig = sns.heatmap(final_score)
        elif mode == "layer-wise":
            x_ = [f"layer_{i}" for i in range(len(model.hidden_states))]
            sns_fig = sns.barplot(x=x_, y=profile_logging, palette="hls")
        plt.savefig(f"./output/{mode}_{file_path}_map.png")
        plt.clf()

def eval(index, model, eval_loader, label_list, file_path, mode="layer-wise", device=args.device):
    # loop_size = len(model.hidden_states) if mode == "layer-wise" else model.num_heads
    with torch.no_grad():
        if mode == "layer-wise":
            # for i in tqdm(range(loop_size)):  # i refers to head or layer
            model.to(device)
            # glue_metric = datasets.load_metric('glue')
            metric = load_metric("seqeval")
            for example_batched in tqdm(eval_loader):
                input_ids = example_batched["input_ids"].to(device)
                attention_mask = example_batched["attention_mask"].to(device)
                labels = example_batched["labels"].int().to(device) # use int()
                outputs = model(input_ids, attention_mask)
                logits = outputs[index] # CLS
                preds = torch.argmax(logits, dim=2).int().to(device) # use int()
                # Remove ignored index (special tokens)
                true_predictions = [
                    [label_list[p] for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(preds, labels)
                ]
                true_labels = [
                    [label_list[l] for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(preds, labels)
                ]
                # glue_metric.add_batch(preds, labels)
                metric.add_batch(predictions=true_predictions, references=true_labels)
            results = metric.compute()
        else:
            # for i in tqdm(range(model.num_heads * len(model.hidden_states))):  # i refers to head * layer
            model.to(device)
            # glue_metric = datasets.load_metric('glue')
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
                # glue_metric.add_batch(preds, labels)
                metric.add_batch(predictions=true_predictions, references=true_labels)
            results = metric.compute()

        print(f"{mode} on {index} {file_path} has been evaluated..")
        return results


def main():
    filePath = get_files_path(f"./dataset/{args.task}")
    for i in range(len(filePath)):
        # set the data loader
        probing_train_dataloader, \
        probing_eval_dataloader, \
        probing_label_list = construct_data_loader(batch_size=args.batch_size, dataset=args.task, filePath=filePath[i],
                                                shuffle=True if not args.no_shuffle else True,
                                                num_workers=args.num_workers)
        # load the model
        model_layer_wise = Bert_4_Classification_Layer_Wise(num_labels=len(probing_label_list))
        model_head_wise = Bert_4_Classification_Head_Wise(num_labels=len(probing_label_list))
        print(f"Start training for Layer-wise on {args.task}")
        train(model_layer_wise, probing_train_dataloader, probing_eval_dataloader, probing_label_list, filePath[i], mode="layer-wise")
        print(f"Start training for Head-wise on {args.task}")
        train(model_head_wise, probing_train_dataloader, probing_eval_dataloader, probing_label_list, filePath[i], mode="head-wise")

if __name__ == "__main__":
    main()

