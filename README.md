# Bachorthesis
Frist how goe the setup. We advice that you should do a conda environment for TabPFN and GRANDE (For this go in TabPFN folder and do pip install . (For GRANDE you go in the Lambda Talent folder and do the same there)).
For the calculation we advicer that you should use a gpu which could acuired by bash gpu.sh.

For usage first determine that you have the ritght conda environment activate. Here we will enumerate which file you shoud use for getting the rigth results. 
changeDeepthAndEstimatorForHP.py : here you can change the HPs depth and estimator for the corresponding objective
ScatterPlotForRobustnessClean.ipynb : here you can see the plotting of the correlation between clean accuracy and robust accuracies (for demonstration you use the python notepad)
1_jovita : is the folder where you can find the GRADE Execution for the dataset, here you will find there different dataset (The content of them is build equally); In on folder for a dataset you go in the execute folder and then you will see three bash scripts. (First train_single.sh, here you train on single objective, Second train_multi.sh you train on multi-objective, thirdly evaluate.sh where you evaluate the model and get the permutation importance)
TabPFN : here is the execution of TabPFN, you must go in examples and then you choose single or multi objective; In single objective you run the script ZCP_Single.py here are the calculation of accuracy and permutation importance; (In folder Multi_Objective you can decide between accuracy and permutation importance; for accuracy you must first generate the data with Datagenerator_for_Linear_Regressor.py then you can execute fast_linear_regressor.py in Linear_Regressor; permutation_Importance you find permuation_importance_multi.py where you can caculate with permutation importance)

Note that for Datagenerator_for_Linear_Regressor.py, permuation_importance_multi.py, ZCP_Single.py you have arguments --dataset (for dataset cifar10,cifar100,imageNet), --objective(clean,fgsm,pgd,apgd,square,clean_fgsm,clean_pgd,clean_apgd,clean_square) and --seed_num (Here you set the seedNumber)
For changeDeepthAndEstimatorForHP.py you have --depth (depth of tree take an integer), --objective (objective like above), --n_estimators(number of the estimators take a integer)

Rest of the folder which has no direkt funktionarlity: in material_for_Thesis you find some pngs which are pre-calculated, LAMDA-TALENT this is the grundgerüst for GRANDE, data/result here are the data stored
