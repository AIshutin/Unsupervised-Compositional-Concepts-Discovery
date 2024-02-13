#!/bin/bash
#SBATCH --job-name=concept-discovery
#SBATCH --error=run-logs/concept-discovery-%j.err
#SBATCH --output=run-logs/concept-discovery-%j.log
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=4:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

module purge
module load Python/Anaconda_v03.2023

source deactivate
source activate decomp2

echo "path is "
pwd
nvidia-smi
accelerate env
which python
python -V
conda list
nvcc --version

bash train_concept_discovery.sh $1