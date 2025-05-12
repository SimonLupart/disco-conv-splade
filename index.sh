#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=index_topiocqa
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-08:00:00
#SBATCH --mem=180gb #120gb
#SBATCH -c 16
#SBATCH --output=slurm_index-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
# conda activate splade
conda activate disco
nvidia-smi

config=disco_topiocqa_mistral_llama.yaml
collection_path=DATA/full_wiki_segments_topiocqa.tsv
index_dir=DATA/topiocqa_index_self

python -m splade.index --config-name=$config \
    init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    data.COLLECTION_PATH="$collection_path" \
    config.index_retrieve_batch_size=128
