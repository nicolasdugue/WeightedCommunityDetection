import os
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-N", "--nodes", help="Nodes number", type=str)
argparser.add_argument("-mk", "--maxk", help="Maximum degree", type=str)
argparser.add_argument("-k", "--avgk", help="Average degree", type=str)
argparser.add_argument("-minc", "--minc", help="Min community size", type=str)
argparser.add_argument("-maxc", "--maxc", help="Max community size", type=str)
argparser.add_argument("-muw", "--muw", help="Muw", type=str)
args = argparser.parse_args()
print(args)
chemin1="lfr_"+args.nodes
chemin2=os.path.join(chemin1,"mk" + args.maxk)
chemin3=os.path.join(chemin2,"k"+args.avgk)
chemin4=os.path.join(chemin3,"muw"+args.muw)
if not os.path.isdir(chemin1):
	os.mkdir(chemin1)
if not os.path.isdir(chemin2):
	os.mkdir(chemin2)
if not os.path.isdir(chemin3):
	os.mkdir(chemin3)
if not os.path.isdir(chemin4):
	os.mkdir(chemin4)
for i in range(50):
	chemin=os.path.join(chemin4, str(i))
	os.mkdir(chemin)
	if args.minc is None or args.maxc is None:
		os.system("./weighted_networks/benchmark -N "+ args.nodes +" -maxk " + args.maxk+" -k "+args.avgk+" -muw "+args.muw)
	else:
		os.system("./weighted_networks/benchmark -N "+ args.nodes +" -maxk " + args.maxk+" -k "+args.avgk+" -muw "+args.muw +" -minc "+ args.minc+" -maxc "+args.maxc)
	os.system("mv community.dat network.dat statistics.dat time_seed.dat "+chemin)
