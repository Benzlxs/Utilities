#!/bin/bash
#SBATCH --account=OD-217715
#SBATCH --job-name=longtail_1
#SBATCH --time=160:00:00 # Max job time
#SBATCH --nodes=1 # Num of nodes
#SBATCH --cpus-per-task=16 # Num of CPU cores
#SBATCH --gres=gpu:1 # Num of GPUs
#SBATCH --mem=256g # RAM size
#SBATCH --mail-type=ALL # Send status to your email if you want. Optional
#SBATCH --error=error.txt # Error infor printed by the job
#SBATCH --output=bs512_ep200_w0001_losshead_only.txt # Output printed by the job
#SBATCH --mail-user=xuesong.li@csiro.au # Optional

module load tmux/3.3a
module load gcc/13.2.0
module load cmake/3.27.6
module load singularity/3.8.7
module load cuda/12.1.0
module load cudnn/8.9.5-cu12

cd /scratch3/li325
source /scratch3/li325/miniconda3/bin/activate
source activate lsc
cd LSC-sam/LSC-sam-protoloss2losses-online

python train.py --config config/iNaturalist2018/iNat18_LSC_Mixup.txt --exp_name 'bs512_ep200_w0001_losshead_only'
#Write the commands here
#sbatch xxx.sh
#squeue -j <JOB_ID>, squeue -u <IDENT>, or slurmtop -u <IDENT>
#scancel -j <JOB_ID>
#scancel -u <IDENT>

