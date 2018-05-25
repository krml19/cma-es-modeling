#!/bin/bash
#SBATCH -p lab-ci,lab-43,lab-44
#SBATCH -x lab-al-9
#SBATCH -c 1 --mem=1475
#SBATCH -t 22:00:00
#SBATCH -Q
date
hostname
echo ok
srun python ./cma-es-modeling/scripts/cmaes.py