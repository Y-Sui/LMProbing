from dataloader import DEFALT_DATASETS
import os

class LoggerConfig:
    def __init__(self):
        self.logging_path = "//home/weicheng/data_interns/yuansui/logging"
        self.checkpoints = "//home/weicheng/data_interns/yuansui/models"
        self.corpus = DEFALT_DATASETS.keys()
        self.output_path = "//home/yuansui/paper-revision/output"
        self.mkdir()

    def mkdir(self):
        os.makedirs(self.logging_path, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        for i in self.corpus:
            os.makedirs(os.path.join(self.logging_path, i), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "M-BERT_layer-wise"), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "XLM-R_layer-wise"), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "M-BERT_head-wise"), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "XLM-R_head-wise"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_head-wise_finetune"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_layer-wise_finetune"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_head-wise_probing"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_layer-wise_probing"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_head-wise_probing"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_head-wise_finetune"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_layer-wise_probing"), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints, i, "XLM-R_layer-wise_probing"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, i), exist_ok=True)