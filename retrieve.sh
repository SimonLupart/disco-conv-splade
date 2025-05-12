#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=retrieve_topiocqa
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-08:00:00
#SBATCH --mem=180gb #120gb
#SBATCH -c 16
#SBATCH --output=slurm_retrieve-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
conda activate disco

nvidia-smi

# HuggingFace Checkpoint
config=disco_topiocqa_mistral_llama.yaml
index_dir=DATA/topiocqa_index
out_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama_out_hf/

python -m splade.retrieve --config-name=$config \
    init_dict.model_type_or_dir_q=slupart/splade-disco-topiocqa-mistral \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    config.out_dir=$out_dir



# Model trained yourself
config=disco_topiocqa_mistral_llama.yaml
ckpt_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama/
index_dir=DATA/topiocqa_index
out_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama_out/

python -m splade.retrieve \
    --config-name=$config \
    config.checkpoint_dir=$ckpt_dir \
    config.index_dir=$index_dir \
    config.out_dir=$out_dir
