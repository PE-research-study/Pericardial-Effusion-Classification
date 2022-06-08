#!/bin/bash
#SBATCH --job-name=PE_VGG8_pre
#SBATCH --partition=short
#SBATCH --mail-user=allenkong1994@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --gres=gpu:P100:1
#SBATCH --mem-per-gpu=16gb
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=PE_VGG8_pre.out
#SBATCH --error=PE_VGG8_pre.err

python PE_VGG8_pre.py