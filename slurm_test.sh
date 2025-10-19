#!/bin/bash
#SBATCH -p gpu                  # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                          # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1                   # Specify number of GPU per task
#SBATCH --ntasks-per-node=4                 # Specify tasks per node
#SBATCH -t 120:00:00                 # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200321                 # Specify project name
#SBATCH -J aig-dec                  # Specify job name

module load Mamba/23.11.0-0         # Load the conda module
conda activate pytorch-2.2.2
generators=("adm" "midjourney" "wukong")
for gen in "${generators[@]}"
do
	echo "Starting test for generator: $gen"
	python test.py --network dual_cnn --transform hog_224 --loader gen_image --generator "$gen" --checkpoint 300
	echo "Finished test for generator: $gen"
	echo "-----------------------------------"
done
