import hydra
from omegaconf import DictConfig
import os
import gc
import torch

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .evaluate import evaluate
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    #if HF: need to udate config.
    if "hf_training" in config and config["hf_training"]:
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None

    model = get_model(config, init_dict)

    batch_size = config["index_retrieve_batch_size"]
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size,
                                        shuffle=False, num_workers=1)

        evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                compute_stats=True, dim_voc=model.output_dim)

        evaluator.batch_retrieve(q_loader, top_k=exp_dict["config"]["top_k"], threshold=exp_dict["config"]["threshold"])
        evaluator = None
        gc.collect()
        torch.cuda.empty_cache()
    evaluate(exp_dict)


if __name__ == "__main__":
    retrieve_evaluate()
