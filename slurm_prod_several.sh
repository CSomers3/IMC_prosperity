#!/bin/bash

#SBATCH --job-name=IMC
#SBATCH --output=outs/%x.o%j.txt
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu_prod
#SBATCH --mail-user=lhotteromain@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

# Load necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0

# Activate anaconda environment
source activate SAGA_env

# Run python script
which python
python local_playground.py
