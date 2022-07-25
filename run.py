import os
import argparse
import datasets
import torch
import torch.nn as nn
from datasets import load_metric
from tqdm import tqdm
from model import Bert_4_Classification_Head_Wise, Bert_4_Classification_Layer_Wise
from dataloader import *

layer_wise_path = "output/bert_classification_layer_wise/"
head_wise_path = "output/bert_classification_head_wise/"

if not os.path.exists(layer_wise_path):
    os.mkdir(layer_wise_path)
if not os.path.exists(head_wise_path):
    os.mkdir(head_wise_path)


# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="SST2", type=str, help="The dataset name, the options can be sst, SST2, etc")
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
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--lr", default=0.0001, type=int)
args = parser.parse_args()

# Setup devices (No distributed training here)
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


def train(model, train_loader, label_list, mode="layer-wise", epochs=args.epochs, device=args.device):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # remove specical token
    optimizer = torch.optim.Adam(
        filter(lambda p:p.requires_grad, model.parameters()), # only update the fc parameters (classifier)
        lr=args.lr,
    )
    loop_size = len(model.hidden_states) if mode == "layer-wise" else model.num_heads
    for i in range(loop_size): # i refers to head or layer
        optimizer.zero_grad() # make sure each layer's optimizer set to zero grad
        for epoch in tqdm(range(1, epochs + 1)):
            for idx, example_batched in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = example_batched["input_ids"].to(device)
                attention_mask = example_batched["attention_mask"].to(device)
                labels = example_batched["labels"].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs[i]
                preds = logits.permute(0,2,1) # adapt to the nn.crossentropy, inputs = [batch_size, nb_classes, *additional_dims]; target in the shape [batch_size, *additional_dims]
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                if idx % 30 == 0:
                    print(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
        output_path = layer_wise_path if mode == "layer-wise" else head_wise_path
        torch.save(model, output_path + f"bert_classification_{i}.bin")

def eval(model, eval_loader, label_list, mode="layer-wise", device=args.device):
    model.to(device)
    loop_size = len(model.hidden_states) if mode == "layer-wise" else model.num_heads
    output_path = layer_wise_path if mode == "layer-wise" else head_wise_path
    final_score = []
    with torch.no_grad():
        for i in range(loop_size):  # i refers to head or layer
            # glue_metric = datasets.load_metric('glue')
            metric = load_metric("seqeval")
            for example_batched in tqdm(eval_loader):
                input_ids = example_batched["input_ids"].to(device)
                attention_mask = example_batched["attention_mask"].to(device)
                labels = example_batched["labels"].int().to(device) # use int()
                outputs = model(input_ids, attention_mask)
                logits = outputs[i] # CLS
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
            final_score.append(results)
    with open(output_path + "bert_classification_layer_wise_logging.txt", "w") as file:
        for i in range(len(final_score)):
            file.write(f"Performance of the {i}th is: "
                       f"precision, {final_score[i]['overall_precision']}, "
                       f"Recall, {final_score[i]['overall_recall']}, "
                       f"F1, {final_score[i]['overall_f1']}, "
                       f"Accuracy, {final_score[i]['overall_accuracy']}" + "\n")



def main():
    # load the model
    model_layer_wise = Bert_4_Classification_Layer_Wise(num_labels=13)
    model_head_wise = Bert_4_Classification_Head_Wise(num_labels=13)
    wnut_train_dataloader, \
    wnut_eval_dataloader, \
    wnut_test_dataloader, \
    wnut_label_list = construct_data_loader(batch_size=args.batch_size,
                                            shuffle=True if not args.no_shuffle else True,
                                            num_workers=args.num_workers)
    # train(model_layer_wise, wnut_train_dataloader, wnut_label_list, mode="layer-wise")
    # eval(model_layer_wise, wnut_eval_dataloader, wnut_label_list, mode="layer-wise")
    train(model_head_wise, wnut_train_dataloader, wnut_label_list, mode="head-wise")
    eval(model_head_wise, wnut_eval_dataloader, wnut_label_list, mode="head-wise")

if __name__ == "__main__":
    main()

