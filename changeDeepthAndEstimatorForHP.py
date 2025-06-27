import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, required=True, help="Depth of the GRANDE model")
parser.add_argument("--n_estimators", type=int, required=True, help="Number of estimators for the GRANDE model")
parser.add_argument("--objective", type=str, required=True, help="Here you can change for the corresponding objective the depth and the estimators")
args = parser.parse_args()


print("Depth:", args.depth)
print("N_estimators:", args.n_estimators)

path = '/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/LAMDA-TALENT/TALENT/configs/default/grande_'+args.objective+'.json'

with open(path, 'r') as f:
    config = json.load(f)

config['grande']['model']['depth'] = args.depth
config['grande']['model']['n_estimators'] = args.n_estimators

with open(path, 'w') as f:
    json.dump(config, f, indent=4)

