import torch
import argparse
import datasets
import torch
import torch.nn as nn
from datasets import load_metric
from tqdm import tqdm
from dataloader import *

from run import args, layer_wise_path, head_wise_path
from model import Bert_4_Classification_Head_Wise, Bert_4_Classification_Layer_Wise
from sklearn.metrics import classification_report


def evaluate(model, eval_loader, label_list, mode="layer-wise", device=args.device):
    loop_size = len(model.hidden_states) if mode == "layer-wise" else model.num_heads
    output_path = layer_wise_path if mode == "layer-wise" else head_wise_path
    final_score = []
    with torch.no_grad():
        for i in range(loop_size):  # i refers to head or layer
            model.load_state_dict(torch.load(output_path + f"bert_classification_{i}.pt"))
            model.to(device)
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
                metric.add_batch(predictions=true_predictions, references=true_labels)
                # for i in range(len(true_predictions)):
                #     tmp_pred = torch.stack(true_predictions[i],0).type(torch.int64)
                #     tmp_label = torch.stack(true_labels[i],0).type(torch.int64)
                #     predictions = nn.functional.one_hot(tmp_pred, len(label_list))
                #     labels = nn.functional.one_hot(tmp_label, len(label_list))
                #     predictions_stack = torch.stack((predictions, predictions))
                #     labels_stack = torch.stack((labels, labels))
            # labels_stack, predictions_stack = labels_stack.to('cpu'), predictions_stack.to('cpu')
            # score = classification_report(labels_stack, predictions_stack, target_names=label_list)
            results = metric.compute()
            # print(results)
            final_score.append(results)
    with open(output_path + "bert_classification_layer_wise_logging_classes.txt", "w") as file:
        for i in range(len(final_score)):
            file.write(f"Performance of the {i}th is: "
                       f"corporation, {final_score[i]['corporation']}, "
                       f"creative-work, {final_score[i]['creative-work']}, "
                       f"group, {final_score[i]['group']}, "
                       f"location, {final_score[i]['location']}, "
                       f"product, {final_score[i]['product']}, "
                       f"Overall precision, {final_score[i]['overall_precision']}, "
                       f"Overall recall, {final_score[i]['overall_recall']}, "
                       f"Overall F1, {final_score[i]['overall_f1']}, "
                       f"Overall accuracy, {final_score[i]['overall_accuracy']}" + "\n")

if __name__ == '__main__':
    wnut_train_dataloader, \
    wnut_eval_dataloader, \
    wnut_test_dataloader, \
    wnut_label_list = construct_data_loader(batch_size=args.batch_size,
                                            shuffle=True if not args.no_shuffle else True,
                                            num_workers=args.num_workers)
    model_layer_wise = Bert_4_Classification_Layer_Wise(num_labels=len(wnut_label_list))
    model_head_wise = Bert_4_Classification_Head_Wise(num_labels=len(wnut_label_list))
    evaluate(model_layer_wise, wnut_eval_dataloader, wnut_label_list, mode="layer-wise")
    evaluate(model_head_wise, wnut_eval_dataloader, wnut_label_list, mode="head-wise")

