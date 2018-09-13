#!/bin/bash
#SBATCH -J runXp4OnLfr_7
#SBATCH --time 06-00
#SBATCH -n 1
#SBATCH -c 20
#SBATCH  --mem-per-cpu 4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.connes@gmail.com


for DIR in `find ../../lfr_5000/ | grep -P "mk.*/k.*/mu.*/\d+$"`
do
  if [ ! -f "$DIR/network.dat" ]; then echo "Network file missing: $DIR/network.dat"; fi;
  if [ ! -f "$DIR/community.dat" ]; then echo "Community file missing: $DIR/community.dat"; fi;
  if [ -f "$DIR/community.dat" ] && [ -f "$DIR/network.dat" ]
  then
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP4_7".log -e $DIR"/XP4_7".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 ResultOn.py $DIR --addAssort &
  else
  echo "pass on $DIR";
fi;
done
wait
