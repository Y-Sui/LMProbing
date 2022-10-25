import json
import logging
import random
import os
import pandas as pd
from config import DataConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def get_files_path(filePath):
    """
    return the file list
    """
    raw_files = os.listdir(filePath)
    file_paths = []
    for i in range(len(raw_files)):
        if raw_files[i].find("train") == -1 and raw_files[i].find("eval") == -1:
           file_paths.append(os.path.join(filePath, raw_files[i]))
    return file_paths


def get_pure_tokens(tokenList, category):
    """
    filter out all the annotated tags to get the pure token list
    """
    filter = ["O-", "X-", f"B-{category}-", f"I-{category}-"]
    if category.islower():
        filter = ["O-", "X-", "B-head-", "B-dependent-"]
    if category == "VP":
        filter = ["O-", "X-", "B-ADVP-", "I-ADVP-", "B-VP-", "I-VP-"]
    if category == "":
        filter = ["O-", "X-", "B-:-", "I-:-"]
    if category == "EX":
        filter = ["O-", "X-", "B-E", "I-E"]
    if category == "TO":
        filter = ["O-", "X-", "B-T", "I-T"]
    tokens = []
    for i in range(len(tokenList)):
        for j in range(len(filter)):
            tokenList[i] = tokenList[i].replace(filter[j], "")
        tokens.append(tokenList[i].split())
    return tokens

def get_tags(tagList, flag="ner"):
    """
    generate the mixed tag list
    """
    if flag == "ner":
        tag_dict = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-MISC": 5, "I-MISC": 6, "B-LOC": 7,
                    "I-LOC": 8}
    if flag == "chunk":
        tag_dict = {"O": 0, "B-ADJP": 1, "I-ADJP": 2, "B-ADVP": 3, "I-ADVP": 4, "B-CONJP": 5, "I-CONJP": 6, "B-INTJ": 7,
                    "I-INTJ": 8, "B-LST": 9, "I-LST": 10, "B-NP": 11, "I-NP": 12, "B-PP": 13, "I-PP": 14, "B-PRT": 15,
                    "I-PRT": 16, "B-SBAR": 17, "I-SBAR": 18, "B-VP": 19, "I-VP": 20}
    tag_keys = list(tag_dict.keys())
    for i in range(len(tagList)):
        tagList[i] = tagList[i].split()
        for j in range(len(tagList[i])):
            for index in range(len(tag_keys)):
                if tagList[i][j].__contains__(f"{tag_keys[index] + '-'}"):
                    tagList[i][j] = tag_dict[f"{tag_keys[index]}"]
                    break
    # how to deal with X
    for i in range(len(tagList)):
        for j in range(len(tagList[i])):
            if isinstance(tagList[i][j], str) and tagList[i][j].__contains__("X-"):
                tagList[i][j] = tagList[i][j - 1]
    return tagList


def get_single_tags(tag_list, category="LOC"):
    """
    generate the single tag list
    """
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
    tag_keys = list(tag_dict.keys())
    for i in range(len(tag_list)):
        tag_list[i] = tag_list[i].split(" ")
        for j in range(len(tag_list[i])):
            for index in range(len(tag_keys)):
                if str(tag_list[i][j]).__contains__(f"{tag_keys[index] + '-'}"):
                    tag_list[i][j] = tag_dict[f"{tag_keys[index]}"]
    # how to deal with X
    for i in range(len(tag_list)):
        for j in range(len(tag_list[i])):
            if isinstance(tag_list[i][j], str) and tag_list[i][j].__contains__("X-"):
                tag_list[i][j] = tag_list[i][j - 1]
    return tag_list

def get_negative_tags(tagList, category="LOC"):
    """
    generate the negative samples (tags)
    """
    negative_tags = get_single_tags(tagList, category)
    for i in range(len(negative_tags)):
        tmp = negative_tags[i].pop(0)
        negative_tags[i].append(tmp)
    return negative_tags


def data_split(full_list, ratio, shuffle=False):
    """
    split the samples into training and evaluation
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return full_list, full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def dump_json(obj, file_name, default=None):
    with open(file_name, 'w') as fw:
        if default is None:
            json.dump(obj, fw)
        else:
            json.dump(obj, fw, default=default)


def main():
    logger = logging.getLogger("Dataset ()")
    logger.info("Preparation Starts..")
    sample_config = DataConfig()
    for task in sample_config.corpus:
        logger.info(f"Start to generate {task} datasets...")
        data_path = os.path.join(sample_config.data_path, f"{task}", "data")
        samples_path = os.path.join(sample_config.data_path, f"{task}", "samples")
        neg_samples_path = os.path.join(sample_config.data_path, f"{task}", "neg_samples")
        os.makedirs(samples_path, exist_ok=True)
        os.makedirs(neg_samples_path, exist_ok=True)
        raw_files = get_files_path(data_path)
        # random_files = []

        for file in range(len(raw_files)):
            pd_reader = pd.read_csv(raw_files[file], header=None)[1].tolist() # only use column 2
            token_list = pd_reader.copy()
            tag_list = pd_reader.copy()
            neg_tag_list = pd_reader.copy()
            category = raw_files[file].split("_")[-1].replace(".csv", "")

            tokens = get_pure_tokens(token_list, category)  # get the pure tokens
            tags = get_single_tags(tag_list, category)  # get the tags
            negative_tags = get_negative_tags(neg_tag_list, category)

            # save json file
            samples, negative_samples = [], []
            for i in range(len(tokens)):
                samples.append({'tokens': tokens[i], f"{task}_tags": tags[i]})
                negative_samples.append({'tokens': tokens[i], f"{task}_tags": negative_tags[i]})

            # down-sampling
            train_samples, eval_samples = data_split(samples, 0.7, shuffle=True)
            train_neg_samples, eval_neg_samples = data_split(negative_samples, 0.7, shuffle=True)
            logger.info(f"Positive sets have {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")
            logger.info(f"Negative sets have {len(train_neg_samples)} training samples and {len(eval_neg_samples)} evaluation samples")
            train_samples, eval_samples = {"data": train_samples}, {"data": eval_samples}
            train_neg_samples, eval_neg_samples = {"data": train_neg_samples}, {"data": eval_neg_samples}

            # save samples for each labels without data_split
            json_file = raw_files[file].split('/')[-1].replace(".csv", "")
            dump_json(train_samples, os.path.join(samples_path, f"{json_file}_train.json"))
            dump_json(eval_samples, os.path.join(samples_path, f"{json_file}_eval.json"))
            dump_json(train_neg_samples, os.path.join(neg_samples_path, f"{json_file}_train.json"))
            dump_json(eval_neg_samples, os.path.join(neg_samples_path, f"{json_file}_eval.json"))
            logger.info(f"The {task}/{json_file} dataset corresponding single files have been saved...")


if __name__ == "__main__":
    main()
