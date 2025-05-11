import json
import os

import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.evaluation.eval import load_and_evaluate
from splade.utils.utils import get_dataset_name
from splade.utils.hydra import hydra_chdir
from .utils.utils import get_initialize_config

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME,version_base="1.2")
def evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict, train=False)

    # for dataset EVAL_QREL_PATH
    # for metric of this qrel
    hydra_chdir(exp_dict)
    eval_qrel_path = exp_dict.data.EVAL_QREL_PATH
    eval_metric = exp_dict.config.eval_metric
    dataset_names = exp_dict.config.retrieval_name
    out_dir = config["out_explicit"]

    res_all_datasets = {}
    res_full_all_datasets = {}
    for i, (qrel_file_path, eval_metrics, dataset_name) in enumerate(zip(eval_qrel_path, eval_metric, dataset_names)):
        if qrel_file_path is not None:
            res = {}
            res_full = {}
            print(eval_metrics, qrel_file_path)
            for metric in eval_metrics:
                qrel_fp=qrel_file_path
                res.update(load_and_evaluate(qrel_file_path=qrel_fp,
                                             run_file_path=config["run_explicit"],
                                             metric=metric,agg=True),)
                res_full.update(load_and_evaluate(qrel_file_path=qrel_fp,
                                             run_file_path=config["run_explicit"],
                                             metric=metric,agg=False),)
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
                res_full_all_datasets[dataset_name].update(res_full)
            else:
                res_all_datasets[dataset_name] = res
                res_full_all_datasets[dataset_name] = res_full
            # out_fp = os.path.join(out_dir, dataset_name, "perf.json")
            # json.dump(res, open(out_fp,"a"))
            out_fp = os.path.join(out_dir, dataset_name, "perf.json")
            json.dump(res, open(out_fp,"a"))
    out_all_fp= os.path.join(out_dir, "perf_all_datasets_solo.json")
    json.dump(res_all_datasets, open(out_all_fp, "a"))
    with open(out_all_fp,"a") as out_all_fp_open:
        out_all_fp_open.write('\n')
    out_all_fp= os.path.join(out_dir, "perf_all_datasets_full.json")
    json.dump(res_full_all_datasets, open(out_all_fp, "a"))
    with open(out_all_fp,"a") as out_all_fp_open:
        out_all_fp_open.write('\n')

    return res_all_datasets

if __name__ == '__main__':
    evaluate()
