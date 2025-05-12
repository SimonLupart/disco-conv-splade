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
#SBATCH --output=/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/slurm_out/train-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
# conda activate splade
conda activate disco

nvidia-smi

port=$(shuf -i 29500-29599 -n 1)


# # distill loss mistral
# # (1) on all 
runpath=/gpfs/work4/0/prjs0871/disco-conv-splade/DATA/topiocqa_distil/distil_run_top_mistral_llama.json
out_dir=mistral_llama
torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train \
    --config-name=disco_topiocqa_mistral_llama.yaml  \
    config.checkpoint_dir="/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/checkpoint_exp/disco_TOPIOCQA_$out_dir/" \
    data.TRAIN.DATASET_PATH=$runpath \
    config.tokenizer_type=Luyu/co-condenser-marco


# # Eval
# bash /projects/0/prjs0871/splade/SLURM/retrieve_topiocqa_splade_arg++.sh /projects/0/prjs0871/splade/EXP/RW/splade++_TOPIOCQA_$out_dir/


