import hydra
from omegaconf import DictConfig
import os

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .models.transformer_rep import Siamese
from .tasks.transformer_evaluator import SparseIndexing, DenseIndexing
from .utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def index(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    #if HF: need to udate config.
    if "hf_training" in config and config["hf_training"]: # and not config.pretrained_no_yamlconfig
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
       print('HF model')

    if "hf" in exp_dict and exp_dict["hf"]["model"]["dense"]:
        # init_dict.pop("agg")
        model = Siamese(**init_dict)
    else:
        model = get_model(config, init_dict)

    d_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["COLLECTION_PATH"], id_style="row_id", filter=("topiocqa" in exp_dict["data"]["COLLECTION_PATH"]), split=config["split"])
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=10, prefetch_factor=4)
    if "hf" in exp_dict and exp_dict["hf"]["model"]["dense"]:
        evaluator = DenseIndexing(model=model, config=config)
    else:
        evaluator = SparseIndexing(model=model, config=config, compute_stats=True)
    evaluator.index(d_loader)


if __name__ == "__main__":
    index()
