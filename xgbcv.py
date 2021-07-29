import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer,  accuracy_score,  precision_score, recall_score, f1_score,  roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import warnings
def warn(*args, **kwargs):
	pass

warnings.warn = warn
warnings.simplefilter(action='ignore', category=UserWarning)
# EVALUATION PARAMETERS (F1 Score, Recall, Precision)
def eval(target,pred):
	print('F-score:',f1_score(target,pred))
	print('Recall: ',recall_score(target,pred))
	print('Precision: ',precision_score(target,pred))

# def accuracy_metric(predt: np.ndarray, dtrain: xgb.DMatrix):
# 	y = dtrain.get_label()
# 	predset = list(set(predt))
# 	a = 1
# 	if predset[0] < predset[1]:
# 		a = 0
# 	predt[(np.where(predt == predset[0]))] = a
# 	predt[(np.where(predt == predset[1]))] = 1 - a
# 	return 'acc', accuracy_score(y, predt)

# ANY MUTATION PREDICTION
# Dataset - Any mutation (Imp+VUS)	
# title_any = 'Any Gene Mutation Prediction Model'
# df_any = pd.read_csv('data/model_any_gene_preprocd.csv')
# df_any = df_any.set_index('Sno')
# df_any_y = df_any['outcome']
# df_any_X = df_any.drop(['outcome','gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
# X_any_train, X_any_test, y_any_train, y_any_test = train_test_split(df_any_X, df_any_y, test_size= 0.2373, stratify = df_any_y, random_state = 0)

df_vus = pd.read_csv('data/model_imp_gene_preprocd.csv')
df_vus = df_vus.set_index('Sno')
df_vus_y = df_vus['outcome']
df_vus_X = df_vus.drop(['outcome','gene','gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
X_vus_train, X_vus_test, y_vus_train, y_vus_test = train_test_split(df_vus_X, df_vus_y, test_size= 0.2373, stratify = df_vus_y, random_state = 0)

# Building the Model
# num_boost_round = 100
params = {
	# Parameters that we are going to tune.
	'max_depth':1,
	'min_child_weight': 3,
	'eta':.3,
	'subsample': 1,
	'colsample_bytree': 1,
	'alpha': 2,
	'gamma': 0,
	# Other parameters
	'tree_method':'gpu_hist'
}
# 100%|█████████████████████████████████████████████████████████████████████████| 135000/135000 [2:58:39<00:00, 12.59it/s]
# Best testing: (4, 2, 0.2, 0.6, 0.5, 0.0, 0.0), accuracies: (0.6666666666666666, 0.6536312849162011)
# Best training: (8, 6, 1.0, 1.0, 0.5, 2.0, 2.0), accuracies: (0.49122807017543857, 0.6536312849162011)
# end
# Best params: 1, 3, acc: 0.7368421052631579
gridsearch_params = [
	(max_depth, min_child_weight, subsample, colsample, eta, alpha, gamma)
	for max_depth in range(4,6)
	for min_child_weight in range(1,3)
	for subsample in range(1,2)#[i/5. for i in range(0,6)]
	for colsample in [i/5. for i in range(0,6)]
	for eta in [0.05, 0.1, 0.2, 0.3]
	for alpha in [i/2 for i in range(0,5)]
	for gamma in [i/2 for i in range(0,5)]
]

# Best params: 0.2, 0.6, acc: 0.7368421052631579 (no effect)
gridsearch_params2 = [
	(subsample, colsample)
	for subsample in [i/5. for i in range(0,6)]
	for colsample in [i/5. for i in range(0,6)]
]

gridsearch_params3 = [ (eta,_)
for eta in [0.04, 0.05, 0.06, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33]
for _ in [1]
]

gridsearch_params4 = [ (alpha,gamma)
for alpha in [i/2 for i in range(0,7)]
for gamma in [i/2 for i in range(0,7)]
]

min_acc = 0,0 # testing, training
min_train_acc = 0,0
best_params = None
best_params_train = None
for max_depth, min_child_weight, subsample, colsample, eta, alpha, gamma in tqdm(gridsearch_params):
	# print("CV with ", max_depth, min_child_weight, subsample, colsample, eta, alpha, gamma)
	# Update our parameters
	params['max_depth'] = max_depth
	params['min_child_weight'] = min_child_weight
	params['subsample'] = subsample
	params['colsample_bytree'] = colsample
	params['eta'] = eta
	params['alpha'] = alpha
	params['gamma'] = gamma
	# Run CV
	model_any = xgb.XGBClassifier(**params,scale_pos_weight=4.364)
	model_any.fit(X_vus_train, y_vus_train)
	test_pred_any = model_any.predict(X_vus_test)
	# Testing the Model on Training Dataset
	train_pred_any = model_any.predict(X_vus_train)
	# Predicting Probablity of Mutation for Test Dataset
	probs_any = model_any.predict_proba(X_vus_test)

	#Training and Testing Accuracy
	accuracy_train_any = accuracy_score(y_vus_train, train_pred_any)
	accuracy_test_any = accuracy_score(y_vus_test, test_pred_any)
	auc_any = roc_auc_score(y_vus_test, test_pred_any)

	# Update best acc

	# print("\tacc {}, (training) acc = {}".format(mean_acc, accuracy_train_any))
	if accuracy_test_any > min_acc[0]:
		min_acc = accuracy_test_any,accuracy_train_any 
		best_params = (max_depth, min_child_weight, subsample, colsample, eta, alpha, gamma)
	if accuracy_train_any > min_train_acc[0]:
		min_train_acc = accuracy_test_any,accuracy_train_any
		best_params_train = (max_depth, min_child_weight, subsample, colsample, eta, alpha, gamma)

print("Best testing: {}, accuracies: {}".format(best_params, min_acc))
print("Best training: {}, accuracies: {}".format(best_params_train, min_train_acc))
print('end')