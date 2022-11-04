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

from model import MBertLayerWise, MBertHeadWise, XLMRHeadWise, XLMRLayerWise, ModelFinetune, ModelProbing
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
        self.logger = logging.getLogger("Train()")

        self.sample_config = LoggerConfig()

        # model checkpoints path
        if self.mode != None:
            self.model_path = os.path.join(self.sample_config.checkpoints, self.corpus, f"{self.model_config}_{self.mode}_{self.fc}")
            self.logging_path = os.path.join(self.sample_config.logging_path, self.corpus, f"{self.model_config}_{self.mode}")
            if self.mode == "layer-wise":
                self.model = MBertLayerWise(args) if args.model_config == "M-BERT" else XLMRLayerWise(args)
                self.loop_size = len(self.model.hidden_states)
            elif self.mode == "head-wise":
                self.model = MBertHeadWise(args) if args.model_config == "M-BERT" else XLMRHeadWise(args)
                self.loop_size = self.model.num_heads * len(self.model.hidden_states)
        else:
            self.model_path = os.path.join(self.sample_config.checkpoints, self.corpus, f"{self.model_config}_{self.fc}")
            self.logging_path = os.path.join(self.sample_config.logging_path, self.corpus, f"{self.model_config}_{self.fc}")

            # load the model according to different purpose
            if self.fc == "finetune":
                self.model = ModelFinetune(args)
            else:
                self.model = ModelProbing(args)
            self.loop_size = 1 # loop 1
            self.logger.info("Not Supported Layer or Head Wise Mode!")

    def train(self, args):
        if self.corpus == "xnli" or self.corpus == "pawsx":
            """
            fine-tune using sequence classification task
            dataset: xnli, pawsx
            """
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
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    if idx % 500 == 0:
                        self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                        wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data})
            try:
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_path, f"{self.corpus}_{self.fc}.pt"))
            except:
                self.logger.info(f"Model has not been saved due to some error")
            self.logger.info(f"{self.corpus}_{self.lang}_{self.fc} has been trained..")
            self.logger.info(f"start to evaluate the model..")
            eval_results = self.eval()
            self.logger.info(f"{self.corpus}/{self.lang} F1, {eval_results}")

        elif self.corpus == "wikiann":
            self.model.to(self.device)
            criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)  # remove special token
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), # only update the fc parameters (classifier)
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
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    if idx % 500 == 0:
                        self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                        wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data})
            try:
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_path, f"{self.corpus}_{self.fc}_{self.lang}_{self.tag_class}.pt")) # xnli_probing_en_NER.pt
            except:
                self.logger.info(f"Model has not been saved due to some error")
            self.logger.info(f"{self.corpus}_{self.lang}_{self.tag_class}_{self.fc} has been trained..")
            self.logger.info(f"start to evaluate the model..")
            eval_results = self.eval()
            self.logger.info(
                f"{self.corpus}-{self.lang}-{self.tag_class} precision: {eval_results['overall_precision']}")
            self.logger.info(f"{self.corpus}-{self.lang}-{self.tag_class} Recall: {eval_results['overall_recall']}")
            self.logger.info(f"{self.corpus}-{self.lang}-{self.tag_class} F1, {eval_results['overall_f1']}")
            self.logger.info(
                f"{self.corpus}-{self.lang}-{self.tag_class} Accuracy, {eval_results['overall_accuracy']}")

            self.report(eval_results)

        return eval_results

    def eval(self):
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
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).int().cpu().numpy() # use int()
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
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).int().to(self.device)  # use int()
                    # self.logger.info(f"label: {labels}")
                    # self.logger.info(f"preds: {preds}")
                    if i == 1:
                        confusion_matrix = confmat(preds, labels)
                        f1 = f1_score(preds, labels, num_classes=self.num_labels).cpu().numpy()
                    else:
                        confusion_matrix += confmat(preds, labels)
                        f1 += f1_score(preds, labels, num_classes=self.num_labels).cpu().numpy()
                self.logger.info(f"(tn, fp; fn, tp): {confusion_matrix.ravel()}")
                results.append(f1 / len(self.dataloader['test']))
                return results
    def report(self, eval_results):
        if self.corpus == "xnli" or self.corpus == "pawsx":
            with open(os.path.join(self.logging_path, f"{self.corpus}_{self.fc}_{self.lang}.csv"), "w") as file:
                self.logger.info(f"{self.lang} F1, {eval_results}")
                wandb.log({ f"valid/{self.lang}/f1": eval_results})
                metric_frame = pd.DataFrame({f"{self.fc}": ["0"], "F1": eval_results})
                metric_frame.to_csv(file, index=False, sep=",")
            self.logger.info(f"{self.corpus}_{self.lang} metrics evaluation has been saved to {self.corpus}_{self.fc}_{self.lang}.csv!")

        elif self.corpus == "wikiann":
            with open(os.path.join(self.logging_path, f"{self.corpus}_{self.fc}_{self.lang}_{self.tag_class}.csv"), "w") as file:
                acc, recall, f1, prec = [], [], [], []
                self.logger.info(f"{self.lang}-{self.tag_class} precision: {eval_results['overall_precision']}")
                self.logger.info(f"{self.lang}-{self.tag_class} Recall: {eval_results['overall_recall']}")
                self.logger.info(f"{self.lang}-{self.tag_class} F1, {eval_results['overall_f1']}")
                self.logger.info(f"{self.lang}-{self.tag_class} Accuracy, {eval_results['overall_accuracy']}")

                wandb.log({
                    f"valid/{self.lang}-{self.tag_class}/acc": eval_results['overall_accuracy'],
                    f"valid/{self.lang}-{self.tag_class}/prec": eval_results['overall_precision'],
                    f"valid/{self.lang}-{self.tag_class}/f1": eval_results['overall_f1'],
                    f"valid/{self.lang}-{self.tag_class}/recall": eval_results['overall_recall']
                })

                acc.append(eval_results['overall_accuracy'])
                recall.append(eval_results['overall_recall'])
                f1.append(eval_results['overall_f1'])
                prec.append(eval_results['overall_precision'])

            metric_frame = pd.DataFrame({f"{self.mode}": ["0"],
                                         "Accuracy": acc, "Precision": prec, "Recall": recall, "F1": f1})
            metric_frame.to_csv(file, index=False, sep=",")
            self.logger.info(f"{self.corpus}-{self.lang}-{self.tag_class} metrics evaluation has been saved to {self.corpus}_{self.fc}_{self.lang}_{self.tag_class}.csv!")


        # Save the profile figure of the output
        if self.profile:
            profile_logging = eval_results
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



