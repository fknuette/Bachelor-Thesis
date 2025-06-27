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

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

class MultiTabPFN(BaseEstimator):
    def __init__(self, model1, model2, X_val, y_val):
        self.model1 = model1
        self.model2 = model2
        self.linearReg = LinearRegression()
        self.X_val = X_val
        self.y_val = y_val

    def fit(self, X, y):
        self.model1.fit(X, y[:,0])
        self.model2.fit(X, y[:,1])
        self.linearReg.fit(self.getFeaturesForLinReg(self.X_val), self.y_val)
        return self

    def predict(self, X_test):
        return self.linearReg.predict(self.getFeaturesForLinReg(X_test))

    def getFeaturesForLinReg(self, X):
        Z1 = self.probabilityDistributionCorrected(self.model1.predict(X, output_type="full"))
        Z2 = self.probabilityDistributionCorrected(self.model2.predict(X, output_type="full"))

        return self.nonlinear_features_transformation(Z1, Z2)

    def probabilityDistributionCorrected(self, dict):
        """
        Returns the probability distribution corrected by the support vectors.
        """
        logits = dict["logits"]  # shape (N, 5000)
        criterion = dict["criterion"]
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        support = (criterion.borders[:-1] + criterion.borders[1:]) / 2
        support = support.to(device)
        logits = logits.to(device)


        probs = torch.softmax(logits, dim=-1)
        weighted_bins = probs * support 
        features = weighted_bins.detach().cpu().numpy()

        return features


    def nonlinear_features_transformation(self, a: np.ndarray, b: np.ndarray):
        eps = 1e-6

        dot = np.sum(a * b, axis=1, keepdims=True)

        # Correct numerical values to avoid NaNs in log
        dot_stable = np.where(dot < eps, eps, dot)

        dot_b2   = np.sum(a * (b ** 2), axis=1, keepdims=True)
        dot_a2   = np.sum((a ** 2) * b, axis=1, keepdims=True)
        log_ab   = np.log(dot_stable)  # keine NaNs mehr

        a_log_b  = np.sum(a * np.log(np.clip(b, eps, None)), axis=1, keepdims=True)
        b_log_a  = np.sum(b * np.log(np.clip(a, eps, None)), axis=1, keepdims=True)

        a2b2     = np.sum((a ** 2) * (b ** 2), axis=1, keepdims=True)
        exp_dot  = np.exp(np.clip(dot, None, 20))
        a_exp_b  = np.sum(a * np.exp(np.clip(b, None, 20)), axis=1, keepdims=True)
        b_exp_a  = np.sum(b * np.exp(np.clip(a, None, 20)), axis=1, keepdims=True)

        return np.concatenate([
            dot, dot_b2, dot_a2, log_ab, a_log_b, b_log_a,
            a2b2, exp_dot, a_exp_b, b_exp_a
        ], axis=1)




parser = argparse.ArgumentParser(description='Args for ZCP evals of Robustness Dataset')
parser.add_argument('--dataset',                 type=str, default='cifar10', help="Choices between cifar10, cifar100 and imageNet")
parser.add_argument("--objective",       type=str, default="square", help="this are the second objective to optimize, choices are fgsm, pgd, square, apgd")
parser.add_argument("--seed_num",               type=int, default=42, help="Seed for random number generation")

args = parser.parse_args()

path =f"/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/data/result/{args.dataset}/dataGeneral.csv"
column = args.objective 

df = pd.read_csv(path)
X = df.drop(['isomorphTo', 'apgd', 'pgd', 'square', 'clean', 'fgsm'], axis=1)
feature_names = X.columns # for the feature importance plot (see below)
X = X.to_numpy()
y = df[['clean', args.objective]].to_numpy()
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

# Initialize a regressor
reg1 = TabPFNRegressor()
reg2 = TabPFNRegressor()
mul = MultiTabPFN(reg1, reg2, X_val, y_val)
mul.fit(X_train, y_train)


# Caculate the permutation importance
perim = permutation_importance(mul, X=X_test, y=y_test, n_repeats=10, random_state=42, scoring='r2')
forest_importances = pd.Series(perim.importances_mean, index=feature_names)

# Do the plot
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=perim.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

# Targetpath
save_dir = os.path.join("plots", args.dataset)
os.makedirs(save_dir, exist_ok=True)  

filename = f"feature_importance_clean_{column}_TabPFN.png"
filepath = os.path.join(save_dir, filename)

plt.savefig(filepath)
print(f"Plot saved as 'feature_importance_clean_{column}.png' is saved")
