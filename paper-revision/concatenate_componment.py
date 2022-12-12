import argparse
import os.path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def concatenate_encoder_classifier(args):
    model_all = AutoModelForSequenceClassification.from_pretrained(args.checkpoints) # all_language
    model_en = AutoModelForSequenceClassification.from_pretrained(args.checkpoints2) # en

    print(model_all.classifier.state_dict())
    # print(model_en.classifier.state_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints)
    config = AutoConfig.from_pretrained(args.checkpoints)

    classifier = model_en.classifier # get
    state_dict = model_all.state_dict()
    append = {}

    for key in model_all.state_dict().keys():
        if key.split(".")[-1] in classifier.state_dict().keys() and key.split(".")[0] == "classifier":
            append[key] = classifier.state_dict()[key.split(".")[-1]]
    state_dict.update(append) # update the classifier parameters
    model_all.load_state_dict(state_dict, strict=False)

    print(model_all.classifier.state_dict())

    model_all.save_pretrained(f"{args.checkpoints}_cat")
    tokenizer.save_pretrained(f"{args.checkpoints}_cat")
    config.save_pretrained(f"{args.checkpoints}_cat")

def test_concatenation(args):
    model_test = AutoModelForSequenceClassification.from_pretrained(f"{args.checkpoints}_cat")
    print(model_test.classifier.state_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="xlm-roberta-base")
    parser.add_argument("--checkpoints2", type=str, default="xlm-roberta-base")
    args = parser.parse_args()
    os.makedirs(f"{args.checkpoints}_cat", exist_ok=True)
    concatenate_encoder_classifier(args)
    test_concatenation(args)

if __name__ == "__main__":
    main()
