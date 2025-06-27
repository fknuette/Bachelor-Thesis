
# Bachelor Thesis – Setup and Execution Guide

## 1. Environment Setup

We recommend creating dedicated **Conda environments** for both *TabPFN* and *GRANDE* to ensure smooth execution.

### TabPFN Installation
Navigate to the `TabPFN` folder and run:
```bash
pip install .
```

### GRANDE Installation
Navigate to the `LAMDA-TALENT` folder and run:
```bash
pip install .
```

### GPU Usage
To use GPU resources, you can activate them via:
```bash
bash gpu.sh
```

---

## 2. Running Experiments

Make sure the correct **Conda environment** is activated before running any scripts. Below is an overview of the key scripts and folders required to reproduce the results.

### a) Tuning Hyperparameters

- **Script**: `changeDeepthAndEstimatorForHP.py`  
  Use this script to modify tree depth and number of estimators for a given objective.  
  **Arguments**:
  - `--depth` (int): Tree depth
  - `--n_estimators` (int): Number of estimators
  - `--objective` (str): One of `clean`, `fgsm`, `pgd`, `apgd`, `square`, `clean_fgsm`, `clean_pgd`, `clean_apgd`, `clean_square`

### b) Robustness Visualization

- **Notebook**: `ScatterPlotForRobustnessClean.ipynb`  
  Plots correlation between clean accuracy and various robustness metrics. You can open this using Jupyter.

### c) GRANDE Execution

- **Folder**: `1_jovita`  
  Contains experiment runs for different datasets.

  Inside each dataset folder, go to `execute/` and use the following bash scripts:
  - `train_single.sh`: Single-objective training
  - `train_multi.sh`: Multi-objective training
  - `evaluate.sh`: Evaluation script including permutation importance calculation

### d) TabPFN Execution

- **Folder**: `TabPFN/examples`

  #### Single Objective:
  - `ZCP_Single.py`: Computes accuracy and permutation importance

  #### Multi Objective:
  - `Datagenerator_for_Linear_Regressor.py`: Generates synthetic data
  - `fast_linear_regressor.py`: Performs accuracy calculation
  - `permuation_importance_multi.py`: Computes permutation importance

  **Common arguments** for `ZCP_Single.py`, `Datagenerator_for_Linear_Regressor.py`, and `permuation_importance_multi.py`:
  - `--dataset`: One of `cifar10`, `cifar100`, `imagenet`
  - `--objective`: As listed above
  - `--seed_num`: Set a seed for reproducibility

---

## 3. Additional Folder Descriptions

- **`material_for_Thesis/`**  
  Contains precomputed PNG images used for visualization in the thesis.

- **`LAMDA-TALENT/`**  
  Core implementation of the GRANDE model.

- **`data/result/`**  
  Stores all data.  

---

Let us know if you need additional guidance or help with setting up your experiments.
