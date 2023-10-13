#!/bin/bash
#SBATCH --time 10:00:00
#SBATCH --mem-per-cpu 4G
#SBATCH -n 2
#SBATCH --job-name error_run${1}_${2}
#SBATCH --time 4:00:00

#method_name =  sys.argv[1]
#function_name = sys.argv[2]
#seed = int(sys.argv[3])
#experiment_name = sys.argv[4]

python bench_run.py $1 $2 $3 $4