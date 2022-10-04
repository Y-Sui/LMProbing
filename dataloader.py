from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset



class Sequence_Classification:
    def __init__(self, args):
        self.task = args.task
        self.ptm = args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.ptm)
        self.max_length = args.max_length

    def load_dataset_huggingface(self, dataset_name="wnut_17"):
        dataset = load_dataset(dataset_name)
        return dataset

    def load_dataset_json(self, filePath="ner"):
        dataset = load_dataset("json", data_files={"train": f"dataset/{filePath}_train.json",
                                                   "validation": f"dataset/{filePath}_eval.json"}, field="data")
        return dataset

    def load_dataset_single_json(self, category="ner", filePath="wsj_annotated_ner_LOC"):
        dataset = load_dataset("json", data_files={"train": f"dataset/{category}/{filePath}_train.json",
                                                   "validation": f"dataset/{category}/{filePath}_eval.json"}, field= "data")
        return dataset

    def construct_data_loader_huggingface(self, batch_size, shuffle=True, num_workers=0):
        """
        construct dataloader from huggingface
        """
        wnut = self.load_dataset_huggingface("wnut_17")
        wnut = wnut.map(self.tokenize_and_align_labels, batched=True)
        # Set the format of your dataset to be compatible with your machine learning framework:
        wnut.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        wnut_train_set, wnut_eval_set, wnut_test_set = wnut["train"], wnut["validation"], wnut["test"]
        wnut_train_dataloader = DataLoader(wnut_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        wnut_eval_dataloader = DataLoader(wnut_eval_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        wnut_test_dataloader = DataLoader(wnut_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        label_list = wnut["train"].features[f"{self.task}_tags"].feature.names

        return wnut_train_dataloader, \
               wnut_eval_dataloader, \
               wnut_test_dataloader, \
               label_list


    def tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(example["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=self.max_length) # is_split_into_words=True, Whether or not the input is already pre-tokenized (e.g., split into words)
        labels = []
        for i, label in enumerate(example[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            # print(word_ids)
            # print(f"index:{i}, label-len:{len(label)}, word-ids-len:{len(word_ids)}")
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    try:
                        label_ids.append(label[word_idx])
                    except:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return  tokenized_inputs


    def construct_data_loader(self, batch_size, idx, file_path, shuffle=True, num_workers=0):
        """
        construct dataloader from custom dataset
        """
        probing_dataset = load_dataset("json", data_files={"train": file_path["train"][idx],
                                                           "validation": file_path["eval"][idx]}, field="data")
        probing_dataset = probing_dataset.map(self.tokenize_and_align_labels, batched=True)

        # Set the format of your dataset to be compatible with your machine learning framework:
        probing_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        probing_train_set, probing_eval_set = probing_dataset["train"], probing_dataset["validation"]
        probing_train_dataloader = DataLoader(probing_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        probing_eval_dataloader = DataLoader(probing_eval_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # Load the label_list (single samples)
        category = file_path["train"][idx].split("_")[-2]
        tag_dict = {"O": 0, f"B-{category}": 1, f"I-{category}": 2}
        if category.islower():
            tag_dict = {"O": 0, "B-head": 1, "B-dependent": 2}
        if category == "VP":
            tag_dict = {"O": 0, "B-ADVP": 1, "I-ADVP": 2, "B-VP":3, "I-VP": 4}
        if category == "":
            tag_dict = {"O": 0, "B-:": 1, "I-:": 2}
        if category == "EX":
            tag_dict = {"O": 0, "B-E:": 1, "I-E:": 2}
        if category == "TO":
            tag_dict = {"O": 0, "B-T:": 1, "I-T:": 2}

        return probing_train_dataloader, probing_eval_dataloader, list(tag_dict.keys())
