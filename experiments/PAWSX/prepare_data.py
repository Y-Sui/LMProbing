from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path

# en fr de es

langs_per_setting = {
    'train': ['en'],
    'test': ['en', 'fr', 'de', 'es']
}

for split in ['train', 'test']:
    out_file = f'experiments/PAWSX/pawsx_{split}_tmp.json'
    if not Path(out_file).is_file():
        Path(out_file).parent.mkdir(exist_ok=True, parents=True)
        datasets = []
        for lang in langs_per_setting[split]:
            dataset = load_dataset('paws-x', lang, split=split)
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
        dataset.to_json(out_file)

    # load and save as tsv in mtdnn format
    df = Dataset.from_json(out_file)
    final_out_file = f'experiments/PAWSX/pawsx_{split}.tsv'
    with open(final_out_file, 'w') as f:
        for i, row in enumerate(df):
            premise = row['sentence1']
            hypo = row['sentence2']
            label = row['label']
            f.write(f'{i}\t{label}\t{premise}\t{hypo}\n')





