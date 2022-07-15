# Code Functionality
The file main.py contains the main of the whole project, therefore, we will need to modify this file in order to run the machine learning algorithms.

Note that the flow of the package is divided into several stages, the code to run each stage is written in a separate python file in the same folder. The stages are as followed:
- Data preprocessing for which the code is written in the file data_preprocess.py
- Generating statistics from the preprocessed data in the file uni_multiStats.py
- Generating features using different feature selection methods in the file featureselection.py
- Other files that also help in doing feature selection includes FCBF.py which helps to do FCBF feature selection method, MRMR.py which helps to do MRMR feature selection method, su_calculation.py which helps to calculate symmetrical uncertainty used in CFS and FCBF method

To run the code, we have different options:
- Running the code of the project in parallel for ranking based feature selection methods or subset based feature selection methods without boosting or bagging, we use the file ranking_subset_run.py
- Running the code in parallel for SFFS based feature selection methods without boosting and bagging, we use the file sfs_run.py
- Running the code in parallel using boosting and bagging we use boost_bag_run.py

The results will generated via:
- Generating scores are written in scoring.py
- Producing heat maps of accuracy and f1 score will be written in the file stats.py

# Add Input File
The required input file is a .csv file where one of the columns contain the dependent variable and the rest of the columns are independent variable.
- The dependent variable has to be binary while the rest of the columns can be nominal or continuous data. 
- An example of such file is SPAQ.csv
- After that, specify target in the file main.py to be the name of the dependent variable in line 6 and 7.
- Change n_seed in line 16 to be the number of times you want to do cross validation.
- Change splits in line 17to be the number of folds you want to do in each cross validation.

# Results
Results will be stored in these following directories:
* /dataparallel stores confusion matrices for each run for each classifier
* /features stores the name of all feature selected for each train-validation set for each combination of feature selection and classifier
* /resultsparallel aggregates all the confusion matrices and accuracy scores for each combination of feature selection and classifier
* /STATS contains: 
  * Baseline.txt: baseline accuracy and baseline F1 scores
  * FeatureSelectionSummary.csv: how many times each feature is chosen across all feature selection methods
  * max_scores_in_summary.csv: maximum accuracy and F1 scores by the best model after cross-validations.
  * noboostbag_Accuracy.png, noboostbag_F1.png: heatmaps generated from max_scores_in_summary.csv


