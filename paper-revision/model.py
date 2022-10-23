import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL_NAMES = {"M-BERT": "bert-base-multilingual-cased",
                       "BERT": "bert-base-uncased",
                       "XLM-R": "xlm-roberta-base"}
DEFAULT_EMBED_SIZE = {"small": 64, "xsmall": 32, "medium": 128, "large": 256}

class LayerWiseConfig(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.classifier_num = args.classifier_num
        self.model = DEFAULT_MODEL_NAMES[args.model_config]
        self.num_labels = args.num_labels

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
        self.embed_size = args.embed_size
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
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, self.embed_size),
            nn.Dropout(0.5),
            nn.Linear(self.embed_size, self.num_labels)
        )