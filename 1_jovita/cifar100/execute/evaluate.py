import torch
from TALENT.model.methods.grande import GRANDEMethod
from argparse import Namespace

from tqdm import tqdm
from TALENT.model.utils import (
    show_results,tune_hyper_parameters,
    get_method,set_seeds
)
from TALENT.model.lib.data import (
    get_dataset
)

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

class SklearnGRANDEWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, grande_method, train_val_data, test_data, info, model_name):
        self.grande_method = grande_method
        self.train_val_data = train_val_data
        self.test_data = test_data
        _, _, self.y = test_data
        self.info = info
        self.model_name = model_name
        self.is_fitted_ = False

    def fit(self, X, y):
        # Instantiate the model (skip actual training)
        # self.grande_method.fit(self.train_val_data, self.info, train=False)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before predict()!")
        input_X = {
            'test' : X
        }
        test_logits = self.grande_method.predict((input_X, None, self.y), self.info, self.model_name, importance=True)
        return test_logits


import argparse

parser = argparse.ArgumentParser(description='Args for ZCP evals of Robustness Dataset')
parser.add_argument('--model',                 type=str, required=True, help="give the name of the model e.g., Epoch100BZ1024-Norm-standard-Nan-mean-new-Cat-indices-Depth-3-Estimators-500")
parser.add_argument("--objective",       type=str, default="clean", help="Objective to optimize, choices are clean, fgsm, pgd, square, apgd, clean_fgsm, clean_pgd, clean_square, clean_apgd")

args = parser.parse_args()


objective = args.objective

if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    trlog_path = f"/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/1_jovita/cifar100/execute/results_model/{objective}-grande/{args.model}/trlog"
    trlog = torch.load(trlog_path, map_location="cpu")
    args = Namespace(**trlog['args'])
    train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)

    json_path = f"/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/1_jovita/cifar100/single/clean/info.json"
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)


    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        set_seeds(args.seed)
        
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        
        # Normal Evaluation Flow
        method.fit(train_val_data, info, train=False)  
        vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
        loss_list.append(vl)
        results_list.append(vres)
        time_list.append(0) #Time cost is not calculated in this case
        # Permutation Importance Evaluation
        # methode = GRANDEMethod(args, True)

        wrapper = SklearnGRANDEWrapper(method, train_val_data, test_data, info, model_name=args.evaluate_option)
        wrapper.fit('dummy_X', 'dummy_y')  # Dummy fit to satisfy sklearn's interface

        from sklearn.inspection import permutation_importance
        modified_test_y_data = method.getTestData()
        perim = permutation_importance(wrapper, X=test_data[0]['test'], y=modified_test_y_data, n_repeats=10, random_state=42, scoring='r2')
    	
        # Save the Plot from the Permutation Importance

        feature_names = [json_data['num_feature_intro'][key] for key in json_data['num_feature_intro']]
        forest_importances = pd.Series(perim.importances_mean, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=perim.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        
        output_dir = f"./plots/{args.dataset}"
        os.makedirs(output_dir, exist_ok=True)


        filename = f"{output_dir}/feature_importance_{seed}_{objective}.png"
        plt.savefig(filename)
        print(f"Plot saved as '{filename}'")
        
        
    
    show_results(args,info, metric_name,loss_list,results_list,time_list)