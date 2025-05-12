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
#SBATCH --output=/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/slurm_out/retrieve-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
conda activate disco

nvidia-smi


config=disco_topiocqa_mistral_llama.yaml #config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e.yaml
index_dir=/gpfs/work4/0/prjs0871/disco-conv-splade/DATA/topiocqa_index

# base_ckpt=/projects/0/prjs0871/splade/EXP/RW/splade++_TOPIOCQA_rwMiLl_10_1000_all__
# echo $1

python -m splade.retrieve --config-name=$config \
    init_dict.model_type_or_dir_q=slupart/splade-disco-topiocqa-mistral \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    config.out_dir="/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/checkpoint_exp/top_out_hf/"
    # config.checkpoint_dir=slupart/splade-disco-topiocqa-mistral \
    # config.checkpoint_dir="$base_ckpt/" \





base_ckpt=/projects/0/prjs0871/splade/EXP/RW/splade++_TOPIOCQA_rwMiLl_10_1000_all__

python -m splade.retrieve --config-name=$config \
    init_dict.model_type_or_dir_q=slupart/splade-disco-topiocqa-mistral \
    config.checkpoint_dir="$base_ckpt/" \
    config.index_dir="$index_dir" \
    config.out_dir="/gpfs/work4/0/prjs0871/disco-conv-splade/EXP/checkpoint_exp/top_out_hf/"
    # config.checkpoint_dir=slupart/splade-disco-topiocqa-mistral \

