import functools
import curiosidade
import torch
import torch.nn
import transformers
import numpy as np
from torch.utils.data import DataLoader
from dataset.sst2.sst2 import SST2

# load the pre-trained model
bert_uncased = transformers.BertForTokenClassification.from_pretrained("../../module/bert-base-uncased")

# set up the probing model

# class ProbingModel(torch.nn.Module):
#     def __init__(self, input_dim: int, out_dim: int):
#         super().__init__()
#         self.params = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(128, out_dim)
#         )
#
#     def forward(self, X):
#         out = X # shape: (batch_size, largest_sequence_length, embed_dim=input_dim)
#         out, _ = out.max(axis=1) # shape: (batch_size, embed_dim=input_dim)
#         out = self.params(out) # shape: (batch_size, output_dim)
#         return out

# # Or, using an available utility function:
ProbingModel = curiosidade.probers.utils.get_probing_model_for_sequences(
    hidden_layer_dims=[128],
    pooling_strategy="max",
    pooling_axis=1,
)

def accuracy_fn(logits, target):
    _, cls_ids = logits.max(axis=-1)
    return {"accuracy": (cls_ids == target).float().mean().item()}

# Prepare dataset
training_data, evaluation_data = SST2("train"), SST2("valid")
probing_dataloader_train = DataLoader(training_data, batch_size=16, shuffle=True)
probing_dataloader_eval = DataLoader(evaluation_data, batch_size=16, shuffle=True)
num_classes = 2

# Set up the probing task
bert_classification_task = curiosidade.ProbingTaskCustom(
    probing_dataloader_train=probing_dataloader_train,
    probing_dataloader_eval=probing_dataloader_eval,
    loss_fn=torch.nn.CrossEntropyLoss(),
    task_name="bert_classification",
    output_dim=num_classes,
    metrics_fn=accuracy_fn,
)

# Set up a ProbingModelFactory, which combines the probing model and the probing task
probing_factory = curiosidade.ProbingModelFactory(
    task=bert_classification_task,
    probing_model_fn=ProbingModel,
    optim_fn=functools.partial(torch.optim.Adam, lr=0.005)
)

# Attach the probing models to the pretrained model layers
prober_container = curiosidade.attach_probers(
    base_model=bert_uncased,
    probing_model_factory=probing_factory,
    modules_to_attach="bert.encoder.layer.\d+.output.dense", # ?
    device="cuda",
)

# Train probing models
probing_results = prober_container.train(num_epochs=10, show_progress_bar="epoch")

# Aggregate results
df_train, df_eval, df_test = probing_results.to_pandas(
    aggregate_by=["batch_index"],
    aggregate_fn=[np.mean, np.std]
)
