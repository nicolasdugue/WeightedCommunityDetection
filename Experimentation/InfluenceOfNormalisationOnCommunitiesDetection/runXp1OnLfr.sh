#!/bin/bash
#SBATCH -J runXp1OnLfr
#SBATCH --time 06-00
#SBATCH -c 20
#SBATCH  --mem-per-cpu 10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.connes@gmail.com


for DIR in `find ../../lfr_5000/ | grep -P ".*\d+$"`
do
  srun -c 20 -n 1  -J "XP1-$DIR" -o $DIR"/XP1".log -e $DIR"/XP1".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 ResultOn.py $DIR &
done
