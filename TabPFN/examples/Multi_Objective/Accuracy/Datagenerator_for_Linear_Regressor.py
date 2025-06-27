#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""
import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
import torch
import argparse


parser = argparse.ArgumentParser(description='Args for ZCP evals of Robustness Dataset')
parser.add_argument('--dataset',                 type=str, default='cifar10', help="Choices between cifar10, cifar100 and imageNet")
parser.add_argument("--objective",       type=str, default="apgd", help="Objective to optimize, choices are clean, fgsm, pgd, square, apgd")
parser.add_argument("--seed_num",               type=int, default=0, help="Seed for random number generation")

args = parser.parse_args()

path =f"/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/data/result/{args.dataset}/dataGeneral.csv"
column = args.objective 

df = pd.read_csv(path)
X = df.drop(['isomorphTo', 'apgd', 'pgd', 'square', 'clean', 'fgsm'], axis=1)
feature_names = X.columns # for the feature importance plot (see below)
X = X.to_numpy()
y = df[column].to_numpy()
# Load data
# X, y = load_diabetes(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=args.seed_num,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,
    random_state=args.seed_num,
)

X_test = np.concatenate([X_val, X_test], axis=0)
y_test = np.concatenate([y_val, y_test], axis=0)

# Initialize a regressor
reg = TabPFNRegressor()
start_time_fit = time.time()
reg.fit(X_train, y_train)
end_time_fit = time.time()
print(f"Fitting time: {end_time_fit - start_time_fit:.4f} seconds")

# Predict a point estimate (using the full)
predictions = reg.predict(X_test, output_type="full")

# 1. Logits und criterion laden
logits = predictions["logits"]              # shape (N, 5000)
criterion = predictions["criterion"]
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# 2. Get the Supportvekotors form the real Domain
support = (criterion.borders[:-1] + criterion.borders[1:]) / 2
support = support.to(device)
logits = logits.to(device)

# 3. Wahrscheinlichkeiten über Softmax berechnen
probs = torch.softmax(logits, dim=-1)       # shape (N, 5000)

# 4. Pro Bin: Wahrscheinlichkeit mal Stützstelle (broadcasted)
weighted_bins = probs * support             # shape (N, 5000)

features = weighted_bins.detach().cpu().numpy()

#import ipdb; ipdb.set_trace()
target = np.array(y_test).reshape(-1, 1)
output = np.concatenate([features, target], axis=1)

save_dir = f"Linear_Regressor/data/{args.seed_num}/{args.dataset}/"
os.makedirs(save_dir, exist_ok=True)  # create if needed

filename = os.path.join(save_dir, f"{column}.npy")
np.save(filename, output)
