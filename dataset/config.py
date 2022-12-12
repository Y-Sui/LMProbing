import os.path

class DataConfig:
    def __init__(self):
        self.data_path = "//home/weicheng/data_interns/yuansui/datasets"
        self.logging_path = "//home/weicheng/data_interns/yuansui/logging"
        self.checkpoints = "//home/weicheng/data_interns/yuansui/models"
        self.corpus = ["ner", "chunk", "pos-fixed", "dependency-fixed"]
        self.output_path = "//home/yuansui/eval-probing/output"
        self.mkdir()

    def mkdir(self):
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.logging_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        for i in self.corpus:
            os.makedirs(os.path.join(self.logging_path, i), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "bert_classification_layer_wise"), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, i, "bert_classification_head_wise"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, i), exist_ok=True)


class ModelConfig:
    def __init__(self, args):
        self.task = args.task




