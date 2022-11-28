import os

from transformers import get_scheduler

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
from evaluate import load

from model import MBertLayerWise, MBertHeadWise, XLMRHeadWise, XLMRLayerWise, ModelFinetune, ModelProbing
from dataloader import DEFALT_DATASETS
from config import LoggerConfig


class TrainerConfig:
    def __init__(self, args, dataloader, label_list):
        self.corpus = args.corpus
        if args.model_config == "M-BERT":
            self.model_config = "mbert"
        elif args.model_config == "XLM-R":
            self.model_config = "xlm"
        self.src = args.src
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
        self.task = args.task

class EvalTrainer(TrainerConfig):
    def __init__(self, args, dataloader, label_list):
        super().__init__(args, dataloader, label_list)
        self.logger = logging.getLogger("Train()")
        self.sample_config = LoggerConfig()

        # model checkpoints path
        self.checkpoint = os.path.join(self.sample_config.checkpoints, f"finetune-{self.src}-{self.model_config}") # /home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert
        self.save_model_path = os.path.join(self.sample_config.checkpoints, self.corpus) # /home/weicheng/data_interns/yuansui/models/ud
        self.logging_path = os.path.join(self.sample_config.logging_path, self.corpus)

        # load the model structure and parameters
        self.model = ModelProbing(args)

    def train(self):
        self.model.to(self.device)
        training_sets = self.dataloader['train']
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),  # only update the fc parameters (classifier)
            lr=self.lr,
        )
        num_training_steps = self.epochs * len(training_sets)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        # if self.corpus == "wikiann" or self.corpus == "ud": # for wikiann, ud-pos, and ud-dependency
        for epoch in range(1, self.epochs + 1):
            for idx, (train_batches, labels) in enumerate(training_sets):
                optimizer.zero_grad()
                input_ids = train_batches["input_ids"].to(self.device)
                attention_mask = train_batches["attention_mask"].to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                preds = outputs.permute(0,2,1).to(self.device)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)  # update progress
                if idx % 50 == 0:
                    self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                    wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data})
            # save state_dict
            checkpoint_name = f"{self.corpus}_{self.lang}_{self.tag_class}.pt" if self.corpus == "wikiann" else f"{self.corpus}_{self.lang}_{self.task}.pt"
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_model_path, checkpoint_name) # xnli_en_NER.pt
            )

        self.logger.info(f"start to evaluate the model..")
        # evaluate
        eval_results = self.eval()
        self.report(eval_results)

    def eval(self):
        logger = logging.getLogger("Eval-probing-evaluation")
        logger.info("Eval() started!")
        with torch.no_grad():
            self.model.to(self.device)
            metric = load_metric("seqeval")
            for (example_batched, labels) in tqdm(self.dataloader["validation"]):
                input_ids = example_batched["input_ids"].to(self.device)
                attention_mask = example_batched["attention_mask"].to(self.device)
                labels = labels.int().cpu().numpy()  # use int()
                outputs = self.model(input_ids, attention_mask)
                logits = outputs
                preds = torch.argmax(logits, dim=-1).int().cpu().numpy() # use int()
                # Remove ignored index (special tokens)
                if self.corpus == "wikiann":
                    true_predictions = [
                        [self.label_list[str(p)] for (p, l) in zip(pred, label) if l != self.pad_token_id]
                        for pred, label in zip(preds, labels)
                    ]
                    true_labels = [
                        [self.label_list[str(l)] for (p, l) in zip(pred, label) if l != self.pad_token_id]
                        for pred, label in zip(preds, labels)
                    ]
                else:
                    true_predictions = [
                        [p for (p, l) in zip(pred, label) if l != self.pad_token_id]
                        for pred, label in zip(preds, labels)
                    ]
                    true_labels = [
                        [l for (p, l) in zip(pred, label) if l != self.pad_token_id]
                        for pred, label in zip(preds, labels)
                    ]
                metric.add_batch(predictions=true_predictions, references=true_labels)
            results = metric.compute()
            return results


    def report(self, eval_results):
        file_path = f"{self.model_config}_{self.src}_{self.lang}_{self.tag_class}.csv" if self.corpus == "wikiann" else f"{self.model_config}_{self.src}_{self.lang}_{self.task}.csv"
        with open(os.path.join(self.logging_path, file_path), "w") as file:
            acc, recall, f1, prec = [], [], [], []
            if self.corpus == "wikiann":
                self.logger.info(f"{self.lang}-{self.tag_class} precision: {eval_results['overall_precision']}")
                self.logger.info(f"{self.lang}-{self.tag_class} Recall: {eval_results['overall_recall']}")
                self.logger.info(f"{self.lang}-{self.tag_class} F1, {eval_results['overall_f1']}")
                self.logger.info(f"{self.lang}-{self.tag_class} Accuracy, {eval_results['overall_accuracy']}")
            else:
                self.logger.info(f"{self.lang} precision: {eval_results['overall_precision']}")
                self.logger.info(f"{self.lang} Recall: {eval_results['overall_recall']}")
                self.logger.info(f"{self.lang} F1, {eval_results['overall_f1']}")
                self.logger.info(f"{self.lang} Accuracy, {eval_results['overall_accuracy']}")

            wandb.log(
                {
                    f"valid/acc": eval_results['overall_accuracy'],
                    f"valid/prec": eval_results['overall_precision'],
                    f"valid/f1": eval_results['overall_f1'],
                    f"valid/recall": eval_results['overall_recall']
                }
            )
            acc.append(eval_results['overall_accuracy'])
            recall.append(eval_results['overall_recall'])
            f1.append(eval_results['overall_f1'])
            prec.append(eval_results['overall_precision'])

            metric_frame = pd.DataFrame({"Accuracy": acc, "Precision": prec, "Recall": recall, "F1": f1})
            metric_frame.to_csv(file, index=False, sep=",")

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

    def tuning(self, config):
        self.model.to(self.device)
        training_sets = self.dataloader['validation']
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),  # only update the fc parameters (classifier)
            lr=config["lr"], # tuning
        )
        num_training_steps = config["epochs"] * len(training_sets)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        if self.corpus == "wikiann":
            for epoch in range(1, config["epochs"] + 1):
                for idx, (train_batches, labels) in enumerate(training_sets):
                    optimizer.zero_grad()
                    input_ids = train_batches["input_ids"].to(self.device)
                    attention_mask = train_batches["attention_mask"].to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                    preds = outputs.permute(0, 2, 1).to(self.device)
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    progress_bar.update(1)  # update progress

                    if idx % 50 == 0:
                        self.logger.info(f"epoch: {epoch}, batch: {idx}, loss: {loss.data}")
                        wandb.log({"epoch": epoch, "batch": idx, "train/loss": loss.data})


