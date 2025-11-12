#!/bin/bash
#SBATCH --job-name=pipeline_NG
#SBATCH --partition=gpu
#SBATCH --output=test_job.out
#SBATCH --error=test_job.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kevin.pasini@irt-systemx.fr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=5G

# (optionnel) Activer l’environnement conda ou venv
#conda activate env_tf

# Exécuter le script Python
python benchmark.py