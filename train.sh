#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=train_topiocqa
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-08:00:00
#SBATCH --mem=180gb #120gb
#SBATCH -c 16
#SBATCH --output=slurm_train-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
conda activate disco

nvidia-smi

port=$(shuf -i 29500-29599 -n 1)

config=disco_topiocqa_mistral_llama.yaml
runpath=DATA/topiocqa_distil/distil_run_top_llama_mistral.json
ckpt_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama/

torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train \
    --config-name=$config  \
    data.TRAIN.DATASET_PATH=$runpath \
    config.checkpoint_dir=$ckpt_dir
