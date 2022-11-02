import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the cuda card 2,3,4,5

from torchmetrics import ConfusionMatrix
from torchmetrics.functional import f1_score

import logging

import numpy as np
import pandas as pd
import wandb
import torch
import seaborn as sns
from datasets import load_metric
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from model import MBertLayerWise, MBertHeadWise, XLMRHeadWise, XLMRLayerWise
from dataloader import DEFALT_DATASETS
from config import LoggerConfig


class TrainerConfig:
    def __init__(self, args, dataloader, label_list):
        self.corpus = args.corpus
        self.model_config = args.model_config
        self.lang = args.lang
        self.epochs = args.epochs
        self.lr = args.lr
        self.profile = args.profile
        self.device = args.device
        self.dataloader = dataloader
        self.label_list = label_list
        self.mode = args.mode
        self.tag_class = args.tag_class
        self.pad_token_id = args.pad_token_id
        self.fc = args.fc
        self.num_labels = args.num_labels

class EvalTrainer(TrainerConfig):
    def __init__(self, args, dataloader, label_list):
        super().__init__(args, dataloader, label_list)
        self.sample_config = LoggerConfig()
        self.logging_path = os.path.join(self.sample_config.logging_path, self.corpus, f"{self.model_config}_{self.mode}")
        self.model_path = os.path.join(self.sample_config.checkpoints, self.corpus, f"{self.model_config}_{self.mode}_{self.fc}")
        self.logger = logging.getLogger("Train()")
        if self.mode == "layer-wise":
            self.model = MBertLayerWise(args) if args.model_config == "M-BERT" else XLMRLayerWise(args)
            self.loop_size = len(self.model.hidden_states)
        elif self.mode == "head-wise":
            self.model = MBertHeadWise(args) if args.model_config == "M-BERT" else XLMRHeadWise(args)
            self.loop_size = self.model.num_heads * len(self.model.hidden_states)
        else:
            self.model = None
            self.loop_size = 0
            self.logger.info("Not Supported Mode!")
    def train(self, args):
        final_score = []
        if self.corpus == "wikiann":
            for i in range(self.loop_size):  # i refers to head or layer
                if self.mode == "layer-wise":
                    self.model = MBertLayerWise(args) if args.model_config == "M-BERT" else XLMRLayerWise(args)
                elif self.mode == "head-wise":
                    self.model = MBertHeadWise(args) if args.model_config == "M-BERT" else XLMRHeadWise(args)
                self.model.to(self.device)
                criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)  # remove special token
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),  # only update the fc parameters (classifier)
                    lr=self.lr,
                )
                optimizer.zero_grad()  # make sure each layer's optimizer set to zero grad
                for epoch in range(1, self.epochs + 1):
                    for idx, (train_batches, labels) in enumerate(self.dataloader['train']):
                        optimizer.zero_grad()
                        input_ids = train_batches["input_ids"].to(self.device)
                        attention_mask = train_batches["attention_mask"].to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                        logits = outputs[i]
                        preds = logits.permute(0, 2, 1).to(self.device)
                        loss = criterion(preds, labels)
                        loss.backward()
                        optimizer.step()
                        if idx % 500 == 0:
                            self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                            wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data, f"{self.mode}-th": i})

                torch.save(self.model.state_dict(), os.path.join(self.model_path, f"{self.fc}_{self.mode}_idx_{i}.pt"))
                self.logger.info(f"{self.mode} {i} on {self.corpus} has been trained..")
                self.logger.info(f"start to evaluate the model..")
                eval_results = self.eval(i)
                self.logger.info(f"{self.corpus}/{self.lang}-{self.tag_class} Performance of the {i}th is:")
                self.logger.info(f"{self.corpus}/{self.lang}-{self.tag_class} precision: {eval_results['overall_precision']}")
                self.logger.info(f"{self.corpus}/{self.lang}-{self.tag_class} Recall: {eval_results['overall_recall']}")
                self.logger.info(f"{self.corpus}/{self.lang}-{self.tag_class} F1, {eval_results['overall_f1']}")
                self.logger.info(f"{self.corpus}/{self.lang}-{self.tag_class} Accuracy, {eval_results['overall_accuracy']}")

                final_score.append(eval_results)

        elif self.corpus == "xnli" or self.corpus == "pawsx":
            for i in range(self.loop_size):  # i refers to head or layer
                if self.mode == "layer-wise":
                    self.model = MBertLayerWise(args) if args.model_config == "M-BERT" else XLMRLayerWise(args)
                elif self.mode == "head-wise":
                    self.model = MBertHeadWise(args) if args.model_config == "M-BERT" else XLMRHeadWise(args)
                self.model.to(self.device)
                criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),  # only update the fc parameters (classifier)
                    lr=self.lr,
                )
                optimizer.zero_grad()  # make sure each layer's optimizer set to zero grad
                for epoch in range(1, self.epochs + 1):
                    for idx, (train_batches, labels) in enumerate(self.dataloader['train']):
                        optimizer.zero_grad()
                        input_ids = train_batches["input_ids"].to(self.device)
                        attention_mask = train_batches["attention_mask"].to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                        logits = outputs[i]
                        # adapt to the nn.crossentropy, inputs = [batch_size, nb_classes, *additional_dims];
                        # target in the shape [batch_size, *additional_dims]
                        preds = logits.permute(0, 2, 1)[:, :, 0].to(self.device) # use CLS to represent all the tokens representation
                        loss = criterion(preds, labels)
                        loss.backward()
                        optimizer.step()
                        if idx % 500 == 0:
                            self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                            wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data, f"{self.mode}-th": i})
                # torch.save(self.model.state_dict(), os.path.join(sample_config.checkpoints, f"{mode}_idx_{i}.pt"))
                self.logger.info(f"{self.mode} {i} on {self.corpus} has been trained..")
                self.logger.info(f"start to evaluate the model..")
                eval_results = self.eval(i)
                self.logger.info(f"{self.corpus}/{self.lang} Performance of the {i}th is:")
                self.logger.info(f"{self.corpus}/{self.lang} F1, {eval_results}")

                final_score.append(eval_results)

        # Save the evaluation
        self.logger.info(f"save the evaluation score to {self.corpus}/{self.lang}-{self.tag_class}.csv")

        if self.corpus == "wikiann":
            profile_logging = []
            try:
                with open(os.path.join(self.logging_path, f"{self.fc}_{self.lang}_{self.tag_class}_{self.mode}.csv"), "w") as file:
                    acc, recall, f1, prec = [], [], [], []
                    for i in range(len(final_score)):
                        self.logger.info(f"{self.lang}-{self.tag_class} Performance of the {i}th is:")
                        self.logger.info(f"{self.lang}-{self.tag_class} precision: {final_score[i]['overall_precision']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} Recall: {final_score[i]['overall_recall']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} F1, {final_score[i]['overall_f1']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} Accuracy, {final_score[i]['overall_accuracy']}")

                        wandb.log({
                            f"valid/{self.lang}-{self.tag_class}/acc": final_score[i]['overall_accuracy'],
                            f"valid/{self.lang}-{self.tag_class}/prec": final_score[i]['overall_precision'],
                            f"valid/{self.lang}-{self.tag_class}/f1": final_score[i]['overall_f1'],
                            f"valid/{self.lang}-{self.tag_class}/recall": final_score[i]['overall_recall'], f"{self.mode}-th": i
                        })

                        acc.append(final_score[i]['overall_accuracy'])
                        recall.append(final_score[i]['overall_recall'])
                        f1.append(final_score[i]['overall_f1'])
                        prec.append(final_score[i]['overall_precision'])
                        # generate the heatmap according to the F1 score
                        profile_logging.append(final_score[i]['overall_f1'])

                    metric_frame = pd.DataFrame({f"{self.mode}": [i for i in range(len(final_score))],
                                                 "Accuracy": acc, "Precision": prec, "Recall": recall, "F1": f1})
                    metric_frame.to_csv(file, index=False, sep=",")
            except:
                path = os.path.join(self.logging_path, f"{self.fc}_{self.lang}_{self.tag_class}_{self.mode}.csv")
                self.logger.info(f"logging save path: {path}")
                with open(path, "w") as file:
                    acc, recall, f1, prec = [], [], [], []
                    for i in range(len(final_score)):
                        self.logger.info(f"{self.lang}-{self.tag_class} Performance of the {i}th is:")
                        self.logger.info(
                            f"{self.lang}-{self.tag_class} precision: {final_score[i]['overall_precision']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} Recall: {final_score[i]['overall_recall']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} F1, {final_score[i]['overall_f1']}")
                        self.logger.info(f"{self.lang}-{self.tag_class} Accuracy, {final_score[i]['overall_accuracy']}")

                        wandb.log({
                            f"valid/{self.lang}-{self.tag_class}/acc": final_score[i]['overall_accuracy'],
                            f"valid/{self.lang}-{self.tag_class}/prec": final_score[i]['overall_precision'],
                            f"valid/{self.lang}-{self.tag_class}/f1": final_score[i]['overall_f1'],
                            f"valid/{self.lang}-{self.tag_class}/recall": final_score[i]['overall_recall'],
                            f"{self.mode}-th": i
                        })

                        acc.append(final_score[i]['overall_accuracy'])
                        recall.append(final_score[i]['overall_recall'])
                        f1.append(final_score[i]['overall_f1'])
                        prec.append(final_score[i]['overall_precision'])
                        # generate the heatmap according to the F1 score
                        profile_logging.append(final_score[i]['overall_f1'])

                    metric_frame = pd.DataFrame({f"{self.mode}": [i for i in range(len(final_score))],
                                                 "Accuracy": acc, "Precision": prec, "Recall": recall, "F1": f1})
                    metric_frame.to_csv(file, index=False, sep=",")
            self.logger.info(f"{self.lang}-{self.tag_class} metrics evaluation has been saved!")

        elif self.corpus == "xnli" or self.corpus == "pawsx":
            profile_logging = final_score

            try:
                with open(os.path.join(self.logging_path, f"{self.fc}_{self.lang}_{self.mode}.csv"), "w") as file:
                    for i in range(len(final_score)):
                        self.logger.info(f"{self.lang} Performance of the {i}th is:")
                        self.logger.info(f"{self.lang} F1, {final_score[i]}")
                        wandb.log({ f"valid/{self.lang}/f1": final_score[i], f"{self.mode}-th": i})

                    metric_frame = pd.DataFrame({f"{self.mode}": [i for i in range(len(final_score))], "F1": final_score})
                    metric_frame.to_csv(file, index=False, sep=",")
            except:
                path = os.path.join(self.logging_path, f"{self.fc}_{self.lang}_{self.mode}.csv")
                self.logger.info(f"logging save path: {path}")
                with open(path, "w") as file:
                    for i in range(len(final_score)):
                        self.logger.info(f"{self.lang} Performance of the {i}th is:")
                        self.logger.info(f"{self.lang} F1, {final_score[i]}")
                        wandb.log({ f"valid/{self.lang}/f1": final_score[i], f"{self.mode}-th": i})

                    metric_frame = pd.DataFrame({f"{self.mode}": [i for i in range(len(final_score))], "F1": final_score})
                    metric_frame.to_csv(file, index=False, sep=",")

            self.logger.info(f"{self.lang} metrics evaluation has been saved!")

        # Save the profile figure of the output
        if self.profile:
            self.logger.info(f"Start to generate the {self.corpus}/{self.lang} profile png..")
            if self.mode == "head-wise":
                final_score = np.reshape(profile_logging, (self.model.num_heads, len(self.model.hidden_states)))
                final_score = pd.DataFrame(final_score, index=[f"head_{i}" for i in range(self.model.num_heads)],
                                           columns=[f"layer_{j}" for j in range(len(self.model.hidden_states))])
                # generate the heatmap according to the F1 score
                sns_fig = sns.heatmap(final_score, vmin=0, vmax=1, cmap="YlGnBu")
            elif self.mode == "layer-wise":
                x_ = [f"{i}" for i in range(len(self.model.hidden_states))]
                sns_fig = sns.barplot(x=x_, y=profile_logging, color="#42b7bd")
                patch_h = [patch.get_height() for patch in sns_fig.patches]
                idx_tallest = np.argmax(patch_h)
                sns_fig.patches[idx_tallest].set_facecolor('#a834a8')
            # plt.savefig(os.path.join(sample_config.output_path, task, f"{label}_{mode}_{file_path}_map.png"))
            plt.clf()

        return final_score

    def eval(self, index):
        logger = logging.getLogger("Eval-probing-evaluation")
        logger.info("Eval() started!")
        with torch.no_grad():
            self.model.to(self.device)
            if self.corpus == "wikiann":
                metric = load_metric("seqeval")
                for (example_batched, labels) in tqdm(self.dataloader["test"]):
                    input_ids = example_batched["input_ids"].to(self.device)
                    attention_mask = example_batched["attention_mask"].to(self.device)
                    labels = labels.int().cpu().numpy()  # use int()
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs[index]  # CLS
                    preds = torch.argmax(logits, dim=2).int().cpu().numpy() # use int()
                    # Remove ignored index (special tokens)
                    true_predictions = [
                        [self.label_list[str(p)] for (p, l) in zip(pred, label) if l != -100]
                        for pred, label in zip(preds, labels)
                    ]
                    true_labels = [
                        [self.label_list[str(l)] for (p, l) in zip(pred, label) if l != -100]
                        for pred, label in zip(preds, labels)
                    ]
                    metric.add_batch(predictions=true_predictions, references=true_labels)
                results = metric.compute()

                logger.info(f"{self.mode} {index} on {self.lang}-{self.tag_class} has been evaluated..")
                return results

            elif self.corpus == "xnli" or self.corpus == "pawsx":
                results = []
                confmat = ConfusionMatrix(num_classes=self.num_labels).to(self.device)
                i = 0
                for (example_batched, labels) in tqdm(self.dataloader["test"]):
                    i += 1
                    input_ids = example_batched["input_ids"].to(self.device)
                    attention_mask = example_batched["attention_mask"].to(self.device)
                    labels = labels.int().to(self.device)  # use int()
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs[index][:,0,:]  # CLS
                    preds = torch.argmax(logits, dim=1).int().to(self.device)  # use int()

                    # self.logger.info(f"label: {labels}")
                    # self.logger.info(f"preds: {preds}")

                    if i == 1:
                        confusion_matrix = confmat(preds, labels)
                        f1 = f1_score(preds, labels, num_classes=self.num_labels)
                    else:
                        confusion_matrix += confmat(preds, labels)
                        f1 += f1_score(preds, labels, num_classes=self.num_labels)
                self.logger.info(f"(tn, fp; fn, tp): {confusion_matrix.ravel()}")
                results.append(f1 / len(self.dataloader['test']))

                logger.info(f"{self.mode} {index} on {self.lang} has been evaluated..")
                return results



