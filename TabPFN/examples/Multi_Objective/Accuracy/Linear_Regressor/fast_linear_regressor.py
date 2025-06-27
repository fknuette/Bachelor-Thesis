import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Transformation
def nonlinear_features_transformation(a, b):
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

# Pfadbasis
base_path = "/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/TabPFN/examples/Multi_Objective/Accuracy/Linear_Regressor/data"

# Parameterlists
datasets = ["cifar10", "cifar100", "imageNet"]
targets = ["fgsm", "pgd", "apgd", "square"]
seeds = [0, 42, 123]

# Collect Results
results = []

# Iterate over Kombination
for dataset in datasets:
    for target in targets:
        for seed in seeds:
            try:
                clean = np.load(f"{base_path}/{seed}/{dataset}/clean.npy")
                other = np.load(f"{base_path}/{seed}/{dataset}/{target}.npy")

                target_clean = clean[:, -1]
                target_other = other[:, -1]
                y = np.stack([target_clean, target_other], axis=-1)
                X_clean = clean[:, :-1]
                X_other = other[:, :-1]
                Z = nonlinear_features_transformation(X_clean, X_other)

                Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.56, random_state=seed)
                reg = LinearRegression()

                t0 = time.time()
                reg.fit(Z_train, y_train)
                t1 = time.time()
                y_pred = reg.predict(Z_test)
                t2 = time.time()

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    "Dataset": dataset,
                    "Target": target,
                    "Seed": seed,
                    "TrainTime_s": t1 - t0,
                    "InferenceTime_s": t2 - t1,
                    "MSE": mse,
                    "MAE": mae,
                    "R2": r2
                })
            except Exception as e:
                print(f"Fehler bei {dataset} - {target} - {seed}: {e}")

# Output DataFrame
df_results = pd.DataFrame(results)
agg_df = df_results.groupby(["Dataset", "Target"]).agg(["mean", "std"])
print(df_results)
print(agg_df)

# Optional speichern:
df_results.to_csv("regression_summary.csv", index=False)
