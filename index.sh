#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=all_topiocqa
##SBATCH --partition gpu
#SBATCH --partition gpu_h100
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --begin=now+1hour
#SBATCH --time=1-08:00:00
#SBATCH --mem=180gb #120gb
#SBATCH -c 16
#SBATCH --output=/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/slurm_out/index-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
# conda activate splade
conda activate disco

nvidia-smi

export SPLADE_CONFIG_NAME="disco_topiocqa_mistral_llama.yaml"

index_dir=/gpfs/work4/0/prjs0871/disco-conv-splade/DATA/topiocqa_index_self
collection_path=/gpfs/work4/0/prjs0871/disco-conv-splade/DATA/full_wiki_segments_topiocqa.tsv

python -m splade.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    data.COLLECTION_PATH="$collection_path" \
    config.index_retrieve_batch_size=128
