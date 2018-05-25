#!/bin/bash
#SBATCH -p lab-ci-student,lab-43-student,lab-44-student
#SBATCH -x lab-al-9-student
#SBATCH -c 1 --mem=1475
#SBATCH -t 22:00:00
#SBATCH -Q
date
hostname
echo ok
srun python ./cma-es-modeling/scripts/cmaes.py
