from __future__ import absolute_import, division, print_function

import datasets


_URL = "data/"
_URLs = {
    "train": _URL + "train.tsv",
    "valid": _URL + "valid.tsv",
    "test": _URL + "test.tsv",
}


class SST2(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="SST2 Dataset",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            supervised_keys=None,
            license="",
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                }
            ),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                text, label = line.split("\t")

                yield idx, {"text": text.strip(), "label": label.strip()}
