#!/bin/bash
#SBATCH -J runXp2OnLfr_7
#SBATCH --time 06-00
#SBATCH -n 1
#SBATCH -c 20
#SBATCH  --mem-per-cpu 4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.connes@gmail.com


LISTGRAPH=`python3 ../Toolbox/representativeSamples.py`
# echo $LISTGRAPH
for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP2_7".log -e $DIR"/XP2_7".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 ResultOn.py $DIR --addAssort &
done

for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP2".log -e $DIR"/XP2".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 ResultOn.py $DIR &
done
wait
