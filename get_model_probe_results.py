from typing import List, Union, Tuple
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model

from data_utils.task_def import EncoderModelType, TaskType
from data_utils.metrics import Metric
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs,
)

def create_heatmap(
    data_csv_path: str = '',
    data_df: Union[pd.DataFrame, None] = None,
    row_labels: List[str] = None,
    column_labels: List[str] = None,
    xaxlabel: str = None,
    yaxlabel: str = None,
    invert_y: bool = False,
    figsize: Tuple[int, int] = (14, 14),
    fontsize: int = 14,
    cmap: str = 'RdYlGn',
    out_file: str= ''
    ):
    """
    General heatmap from data.
    """
    # read data if dataframe not directly supplied.
    if data_df is None:
        data_df = pd.read_csv(data_csv_path, index_col=0)
        assert len(out_file) > 0, f'invalid csv: {data_csv_path}'
    
    plt.figure(figsize=figsize)
    annot_kws = {
        "fontsize":fontsize,
    }
    heatmap = sns.heatmap(
        data_df.to_numpy(),
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        cmap=cmap)

    if invert_y:
        heatmap.invert_yaxis()

    heatmap.set_ylabel(yaxlabel, fontsize=fontsize)
    heatmap.set_xlabel(xaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def, device_id):
    test_data_set = SingleTaskDataset(
        path=data_path,
        is_train=False,
        maxlen=max_seq_len,
        task_id=0,
        task_def=task_def
    )

    collater = Collater(is_train=False, encoder_type=encoder_type)

    test_data = DataLoader(
        test_data_set,
        batch_size=batch_size,
        collate_fn=collater.collate_fn,
        pin_memory=device_id > 0
    )

    return test_data

def construct_model(task: Experiment, setting: LingualSetting, device_id: int):
    if setting is not LingualSetting.BASE:
        task_def_path = Path('experiments').joinpath(task.name, 'task_def.yaml')
        task_name = task.name.lower()
    else:
        # dummy
        task_def_path = Path('experiments').joinpath('NLI', 'task_def.yaml')
        task_name = 'nli'

    task_defs = TaskDefs(task_def_path)
    task_def = task_defs.get_task_def(task_name)

    # load model
    if setting is not LingualSetting.BASE:
        checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_{setting.name.lower()}')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        assert os.path.exists(checkpoint), checkpoint
    else:
        # dummy.
        checkpoint_dir = Path('checkpoint').joinpath(f'NER_multi')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]

    state_dict = torch.load(checkpoint)
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id
    config['head_probe'] = False

    if 'optimizer' in state_dict:
        del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id])
    if setting is LingualSetting.BASE:
        return model

    # scoring_list classification head doesn't matter because we're just taking
    # the model probe outputs.
    if 'scoring_list.0.weight' in state_dict['state']:
        state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
        state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    model.load_state_dict(state_dict)
    return model

def get_acc(model, test_data, metric_meta, device_id, model_probe):
    with torch.no_grad():
        model.network.eval()
        model.network.to(device_id)
        
        results = eval_model(
            model,
            test_data,
            task_type=TaskType.Classification,
            metric_meta=metric_meta,
            device=device_id,
            with_label=True,
            model_probe=model_probe
        )
    metrics = results[0]
    metric_name = metric_meta[0].name
    return metrics[metric_name]

def evaluate_model_probe(
    downstream_task: Experiment,
    finetuned_task: Union[Experiment, None],
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting,
    metric: str,
    batch_size: int=8,
    max_seq_len: int=512,
    device_id: int=0):

    """
    Evaluate model probe for a model finetuned on finetuned_task on a downstream_task.
    """
    task_def_path = Path('experiments').joinpath(
        downstream_task.name,
        'task_def.yaml'
    )
    task_def = TaskDefs(task_def_path).get_task_def(downstream_task.name.lower())

    # test data is always multilingual.
    data_path = Path('experiments').joinpath(
        downstream_task.name,
        'multi',
        'bert-base-multilingual-cased',
        f'{downstream_task.name.lower()}_test.json'
    )
    print(f'data from {data_path}')

    test_data = build_dataset(
        data_path,
        EncoderModelType.BERT,
        batch_size,
        max_seq_len,
        task_def,
        device_id)

    model = construct_model(
        finetuned_task,
        finetuned_setting,
        device_id)
    
    metric_meta = (Metric[metric.upper()],)
    
    if finetuned_task is not None:
        print(f'\n{finetuned_task.name}_{finetuned_setting.name.lower()} -> {downstream_task.name}, {probe_setting.name.lower()}_head_training')
    else:
        print(f'\nmBERT -> {downstream_task.name}, {probe_setting.name.lower()}')
    
    # load state dict for the attention head
    if finetuned_task is not None:
        state_dict_for_head = Path('checkpoint').joinpath(
            'full_model_probe',
            f'{probe_setting.name.lower()}_head_training',
            finetuned_task.name,
            finetuned_setting.name.lower(),
            downstream_task.name,
        )
    else:
        state_dict_for_head = Path('checkpoint').joinpath(
            'full_model_probe',
            f'{probe_setting.name.lower()}_head_training',
            'mBERT',
            downstream_task.name,
        )

    print(f'loading from {state_dict_for_head}')
    state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
    state_dict_for_head = torch.load(state_dict_for_head)['state']

    # then attach the probing layer
    model.attach_model_probe(task_def.n_class)

    # get the layer and check
    layer = model.network.get_pooler_layer()
    assert hasattr(layer, 'model_probe_head')

    # and load (put it on same device)
    weight = state_dict_for_head[f'bert.pooler.model_probe_head.weight']
    bias = state_dict_for_head[f'bert.pooler.model_probe_head.bias']
    
    # weight_save_path = Path(f'debug/{probe_setting.name.lower()}_head_training/{finetuned_task.name}/{downstream_task.name}.pt')
    # weight_save_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(weight, weight_save_path)

    layer.model_probe_head.weight = nn.Parameter(weight.to(device_id))
    layer.model_probe_head.bias = nn.Parameter(bias.to(device_id))

    # compute acc and save
    acc = get_acc(model, test_data, metric_meta, device_id, model_probe=True)
        
    return acc

def combine_all_model_probe_scores():
    combined_results = None

    for task in list(Experiment):
        for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:  
            result_for_task = f'model_probe_outputs/cross_training/{task.name}_{setting.name.lower()}/evaluation_results.csv'
            result_for_task = pd.read_csv(result_for_task, index_col=0)

            if combined_results is None:
                combined_results = result_for_task
            else:
                combined_results = pd.concat([combined_results, result_for_task], axis=0)
    
    combined_results.to_csv('model_probe_outputs/final_result.csv')
    create_heatmap(
        data_df=combined_results,
        row_labels=list(combined_results.index),
        column_labels=list(combined_results.columns),
        xaxlabel='task',
        yaxlabel='model',
        out_file=f'model_probe_outputs/final_result'
    )

def get_model_probe_final_score(
    finetuned_task: Experiment,
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting):

    final_results_out_file = Path(f'model_probe_outputs').joinpath(
        f'{probe_setting.name.lower()}_training',
        f'{finetuned_task.name}_{finetuned_setting.name.lower()}',
        'evaluation_results.csv')

    result_path_for_finetuned_model = final_results_out_file.parent.joinpath('results.csv')
    
    result_path_for_mBERT = Path(f'model_probe_outputs').joinpath(
            f'{probe_setting.name.lower()}_training',
            f'mBERT',
            'results.csv')
    
    finetuned_results = pd.read_csv(result_path_for_finetuned_model, index_col=0)
    mBERT_results = pd.read_csv(result_path_for_mBERT, index_col=0)

    print(finetuned_results)
    print(mBERT_results)

    final_results = pd.DataFrame(finetuned_results.values - mBERT_results.values)
    final_results.index = finetuned_results.index
    final_results.columns = finetuned_results.columns
    final_results.to_csv(final_results_out_file)

    create_heatmap(
        data_df=final_results,
        row_labels=finetuned_results.index,
        column_labels=finetuned_results.columns,
        xaxlabel='task',
        yaxlabel='model',
        out_file=final_results_out_file.with_suffix('')
    )


def get_model_probe_scores(
    finetuned_task: Experiment,
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting,
    metric: str,
    device_id: int,
    batch_size: int = 8,
    max_seq_len: int = 512):
    
    if finetuned_setting is LingualSetting.BASE:
        model_name = 'mBERT'
        finetuned_task = None
    else:
        model_name = f'{finetuned_task.name}_{finetuned_setting.name.lower()}'

    results_out_file = Path(f'model_probe_outputs').joinpath(
        f'{probe_setting.name.lower()}_training',
        model_name,
        'results.csv')

    if results_out_file.is_file():
        print(f'{results_out_file} already exists.')
        return
    else:
        print(results_out_file.parent)
        results_out_file.parent.mkdir(parents=True, exist_ok=True)
    
    tasks = list(Experiment)
    tasks.remove(Experiment.NLI)

    results = pd.DataFrame(np.zeros((1, len(tasks))))
    results.index = [model_name]
    results.columns = [task.name for task in tasks]
    
    for downstream_task in tasks:
        acc = evaluate_model_probe(
            downstream_task,
            finetuned_task,
            finetuned_setting,
            probe_setting,
            metric,
            batch_size,
            max_seq_len,
            device_id)

        if finetuned_setting is LingualSetting.BASE:
            results.loc[f'mBERT', downstream_task.name] = acc
        else:
            results.loc[f'{finetuned_task.name}_{finetuned_setting.name.lower()}', downstream_task.name] = acc
        
    results.to_csv(results_out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--finetuned_task', type=str, default='NLI')
    parser.add_argument('--finetuned_setting', type=str, default='multi')
    parser.add_argument('--probe_setting', type=str, default='cross')
    parser.add_argument('--metric', type=str, default='F1MAC')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    args = parser.parse_args()
    
    # get_model_probe_scores(
    #     Experiment[args.finetuned_task],
    #     LingualSetting[args.finetuned_setting.upper()],
    #     LingualSetting[args.probe_setting.upper()],
    #     args.metric,
    #     args.device_id,
    #     args.batch_size,
    #     args.max_seq_len
    # )

    # get_model_probe_final_score(
    #     Experiment[args.finetuned_task],
    #     LingualSetting[args.finetuned_setting.upper()],
    #     LingualSetting[args.probe_setting.upper()]
    # )

    # combine_all_model_probe_scores()
