import os
import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification

DEFAULT_MODEL_NAMES = {"M-BERT": "bert-base-multilingual-uncased",
                       "BERT": "bert-base-uncased",
                       "XLM-R": "xlm-roberta-large"}
DEFAULT_EMBED_SIZE = {"small": 64, "xsmall": 32, "medium": 128, "large": 256}

# get the latest checkpoints
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class CommonConfig(nn.Module):
    """
    Finetune MBert model using xnli and paswx (text/sequence classification task)
    """
    def __init__(self, args):
        super().__init__()
        self.corpus = args.corpus
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.embed_size = DEFAULT_EMBED_SIZE[args.embed_size]
        self.classifier_num = args.classifier_num
        self.model = DEFAULT_MODEL_NAMES[args.model_config]
        self.num_labels = args.num_labels
        self.fc = args.fc
        self.checkpoints = args.checkpoints
        self.config = AutoConfig.from_pretrained(self.model)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=self.num_labels)

class ModelFinetune(CommonConfig):
    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        return backbone

class ModelProbing(CommonConfig):
    def __init__(self, args):
        super().__init__(args)
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(args.checkpoints):
            last_checkpoint = get_last_checkpoint(args.checkpoints)
            if last_checkpoint is None and len(os.listdir(args.checkpoints)) > 0:
                raise ValueError(
                    f"Output directory ({args.checkpoints}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                print(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        self.backbone = AutoModel.from_pretrained(last_checkpoint)
        # ignore_mismatched_sizes will load the embedding and encoding layers of your model, but will randomly initialize the classification head
        # very interesting parameter, however this approach does not provide any meaningful results due to random init for the new labels
        # self.backbone = AutoModelForTokenClassification.from_pretrained(last_checkpoint, num_labels = self.num_labels, ignore_mismatched_sizes=True) # using TokenClassification instead of SequenceClassification
        self.probe = nn.Linear(self.backbone.hidden_size, self.num_labels) # probing layer

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        return self.probe(backbone)

class LayerWiseConfig(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.embed_size = DEFAULT_EMBED_SIZE[args.embed_size]
        self.classifier_num = args.classifier_num
        self.model = DEFAULT_MODEL_NAMES[args.model_config]
        self.num_labels = args.num_labels
        self.fc = args.fc

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)  # output all the hidden states rather than the last layer
        layers_logits = list(backbone.hidden_states)  # save each layer's output
        for layer_idx in range(len(layers_logits)):
            layers_logits[layer_idx] = self.classifier(layers_logits[layer_idx])
        return layers_logits

    def _get_last_hidden_state_size(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states
        return backbone.hidden_states[0].shape[-1]

class HeadWiseConfig(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.embed_size = DEFAULT_EMBED_SIZE[args.embed_size]
        self.classifier_num = args.classifier_num
        self.model = DEFAULT_MODEL_NAMES[args.model_config]
        self.num_labels = args.num_labels

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)  # output all the hidden states rather than the last layer
        layers = list(backbone.hidden_states)
        heads_logits = []
        for layer_idx in range(len(layers)):
            for head_idx in range(self.num_heads):
                split_heads = layers[layer_idx].chunk(self.num_heads, 2) # split 768 into 12 sections (heads)
                heads_logits.append(self.classifier(split_heads[head_idx]))
        return heads_logits

    def _get_last_hidden_state_size(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states
        return backbone.hidden_states[0].shape[-1]

class MBertLayerWise(LayerWiseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = AutoModel.from_pretrained(self.model)
        if self.fc == "probing":
            for p in self.backbone.parameters():
                p.requires_grad = False  # freeze the backbone model
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, self.embed_size),
            nn.Dropout(0.5),
            nn.Linear(self.embed_size, self.num_labels)
        )

class MBertHeadWise(HeadWiseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = AutoModel.from_pretrained(self.model)
        self.num_heads = self.backbone.config.num_attention_heads
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        if self.fc == "probing":
            for p in self.backbone.parameters():
                p.requires_grad = False  # freeze the backbone model
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, self.embed_size),
            nn.Dropout(0.5),
            nn.Linear(self.embed_size, self.num_labels)
        )

class XLMRLayerWise(LayerWiseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = AutoModel.from_pretrained(self.model)
        if self.fc == "probing":
            for p in self.backbone.parameters():
                p.requires_grad = False  # freeze the backbone model
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, self.embed_size),
            nn.Dropout(0.5),
            nn.Linear(self.embed_size, self.num_labels)
        )

class XLMRHeadWise(HeadWiseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = AutoModel.from_pretrained(self.model)
        self.num_heads = self.backbone.config.num_attention_heads
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        if self.fc == "probing":
            for p in self.backbone.parameters():
                p.requires_grad = False  # freeze the backbone model
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, self.embed_size),
            nn.Dropout(0.5),
            nn.Linear(self.embed_size, self.num_labels)
        )