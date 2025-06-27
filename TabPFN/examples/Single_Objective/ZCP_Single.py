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
parser.add_argument("--objective",       type=str, default="clean", help="Objective to optimize, choices are clean, fgsm, pgd, square, apgd")
parser.add_argument("--seed_num",               type=int, default=0, help="Seed for random number generation")

args = parser.parse_args()

# General Deklaration
dataset = args.dataset 
column = args.objective

# Testing if permutation Importance works properly
def swap_columns(df, col1, col2):
    cols = list(df.columns)
    i, j = cols.index(col1), cols.index(col2)
    cols[i], cols[j] = cols[j], cols[i]
    return df[cols]

path =f"/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/data/result/{dataset}/dataGeneral.csv"
df = pd.read_csv(path)
X = df.drop(['isomorphTo', 'apgd', 'pgd', 'square', 'clean', 'fgsm'], axis=1)

# concrete test
# X = swap_columns(X, 'hessian', 'fisher')
# X = swap_columns(X, 'jacob_fro', 'grasp')

feature_names = X.columns # for the feature importance plot (see below)
X = X.to_numpy()
y = df[column].to_numpy()
# Load data
# X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=args.seed_num,
)

# Initialize a regressor
reg = TabPFNRegressor()
start_time_fit = time.time()
reg.fit(X_train, y_train)
end_time_fit = time.time()
print(f"Fitting time: {end_time_fit - start_time_fit:.4f} seconds")

# Predict a point estimate (using the mean)
start_time = time.time()
predictions = reg.predict(X_test)
end_time = time.time()
print(f"Prediction time: {end_time - start_time:.4f} seconds")

# Metrics Output
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))


# Permutation Importance
perim = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=args.seed_num, scoring='r2')

# Save the Plot from the Permutation Importance
forest_importances = pd.Series(perim.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=perim.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()


output_dir = f"./plots/{args.dataset}"
os.makedirs(output_dir, exist_ok=True)


filename = f"{output_dir}/feature_importance_{column}_TabPFN.png"
plt.savefig(filename)
print(f"Plot saved as '{filename}'")

