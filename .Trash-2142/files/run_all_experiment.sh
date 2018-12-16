#!/bin/bash
#SBATCH -J runXp1OnLfr
#SBATCH --time 06-00
#SBATCH -n 1
#SBATCH -c 20
#SBATCH  --mem-per-cpu 4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.connes@gmail.com


LISTGRAPH=`python3 Experimentation/Toolbox/representativeSamples.py`
# echo $LISTGRAPH

for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP1".log -e $DIR"/XP1".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/InfluenceOfNormalisationOnCommunitiesDetection/ResultOn.py $DIR &
done


for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP2_7".log -e $DIR"/XP2_7".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/EdgesClassification/ResultOn.py $DIR --addAssort &
done

for DIR in LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP2".log -e $DIR"/XP2".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/EdgesClassification/ResultOn.py $DIR &
done


for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP3_7".log -e $DIR"/XP3_7".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/EdgesClassificationWithSavedModel/ResultOn.py $DIR --addAssort &
done

for DIR in LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP3".log -e $DIR"/XP3".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/EdgesClassificationWithSavedModel/ResultOn.py $DIR &
done


for DIR in $LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP4_7".log -e $DIR"/XP4_7".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/ImprovePLP/ResultOn.py $DIR --addAssort &
done

for DIR in LISTGRAPH
do
  echo "computing on $DIR";
  Name=`expr match "$DIR" ".*/lfr_5000/\(mk[0-9]*/k[0-9]*/muw[0-9]*\.[0-9]*/[0-9]*\)$"`;
  echo $Name;
  srun -c 5  -J "$Name" -o $DIR"/XP4".log -e $DIR"/XP4".err --time 04-00 --mail-type=ALL --mail-user=victor.connes@gmail.com \
      python3 Experimentation/ImprovePLP/ResultOn.py $DIR &
done
wait
