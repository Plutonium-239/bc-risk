import pandas as pd
import xgboost as xgb
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




# ANY MUTATION PREDICTION
# Dataset - Any mutation (Imp+VUS)	
title_any = 'Any Gene Mutation Prediction Model'
df_any = pd.read_csv('data/model_any_gene_preprocd.csv')
df_any = df_any.set_index('Sno')
df_any_y = df_any['outcome']
df_any_X = df_any.drop(['outcome','gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
X_any_train, X_any_test, y_any_train, y_any_test = train_test_split(df_any_X, df_any_y, test_size= 0.2373, stratify = df_any_y, random_state = 0)

# Setting up DMatrices
dtrain_any = xgb.DMatrix(X_any_train, label=y_any_train)
weights_any = []
weight_dict_any = {1:0.71,0:1}
for i in y_any_train:
	weights_any.append(weight_dict_any[i])

# Building the Model
model_any = xgb.XGBClassifier(scale_pos_weight=0.7102, verbosity=0)

#Crossvalidation
# params = {"objective":"binary:logistic",'colsample_bytree': 0.7,'learning_rate': 0.1,
# 	 'max_depth': 5, 'alpha': 0.5, 'tree_method':'gpu_hist','min_child_weight':2, 'gamma':2,'n_estimators':200}

# xgb_cv = xgb.cv(dtrain=dtrain_any, params=params, nfold=5,
# 					num_boost_round=50, early_stopping_rounds=10, metrics="auc",stratified=True, 
# 					as_pandas=True, seed=1, verbose_eval=True, show_stdv= True)

distributions = {
		"learning_rate"    : [0.075,0.10,1.25] ,
		"max_depth"        : [4,5,6],
		"colsample_bytree" : [0.4,0.5,0.6],
		"n_estimators"     : [100,150,200],
		"alpha"            : [0.9,1],
		"gamma"	           : [1.5,2,2.5],
		"min_child_weight" : [0,1,2]      
}
scorers = {
			'f1_score':make_scorer(f1_score),
			'accuracy_score': make_scorer(accuracy_score),
			'auc': make_scorer(roc_auc_score)
		  }

rskfold = RepeatedStratifiedKFold(n_splits=5, random_state=1,n_repeats=3)
# clf = RandomizedSearchCV(model_any,random_state=0, param_distributions=distributions, 
# 			cv=rskfold.split(X_any_train, y_any_train), scoring=scorers, n_jobs=-1, refit='auc',verbose=0)
clf = GridSearchCV(model_any, param_grid=distributions, 
			cv=rskfold.split(X_any_train, y_any_train), scoring=scorers, n_jobs=-1, refit='auc', verbose=1)
search = clf.fit(X_any_train, y_any_train, sample_weight=weights_any)
f = open('CV results.txt', 'a')
f.write('GridSearchCV without feature extraction')
f.write('Parameters: ' + str(search.best_params_))
f.write('Score: ' + str(search.best_score_))
p = search.best_params_

# Using the best parameters obtained after cross-validation
model_any = xgb.XGBClassifier(scale_pos_weight=0.7102,colsample_bytree=p['colsample_bytree'], 
	learning_rate=p['learning_rate'], max_depth=p['max_depth'], alpha=p['alpha'],
	min_child_weight=p['min_child_weight'],gamma=p['gamma'],n_estimators=p['n_estimators'])

# Training the Model
model_any.fit(X_any_train, y_any_train, sample_weight=weights_any)

# Testing the Model on Test Dataset
test_pred_any = model_any.predict(X_any_test)

# Testing the Model on Training Dataset
train_pred_any = model_any.predict(X_any_train)

# Predicting Probablity of Mutation for Test Dataset
probs_any = model_any.predict_proba(X_any_test)

#Training and Testing Accuracy
accuracy_train_any = accuracy_score(y_any_train, train_pred_any)
accuracy_test_any = accuracy_score(y_any_test, test_pred_any)
auc_any = roc_auc_score(y_any_test, test_pred_any)

f.write("Training Accuracy: %.2f%%" % (accuracy_train_any * 100.0))
f.write("Test Accuracy: %.2f%%" % (accuracy_test_any * 100.0))
f.write(str(auc_any))

f.close()
f = open('CV results 2.txt', 'a')


### GridSearch after PCA
# X_any_train.drop('Sno')
# X_any_test.drop('Sno')
pca = PCA(n_components=20)
pca_res = pca.fit_transform(X_any_train)
pca_test = pca.fit_transform(X_any_test)
distributions = {
		"learning_rate"    : [0.1, 0.2, 0.3] ,
		"max_depth"        : [3,4,5],
		"colsample_bytree" : [0.2,0.5,0.8],
		"n_estimators"     : [150,300,500],
		"alpha"            : [0.5,0.9,1],
		"gamma"	           : [0,2,4],
		"min_child_weight" : [1,3,5]      
}
model_any = xgb.XGBClassifier(scale_pos_weight=0.7102, verbosity=0)
clf = GridSearchCV(model_any, param_grid=distributions, 
			cv=rskfold.split(pca_res, y_any_train), scoring=scorers, n_jobs=-1, refit='auc',verbose=1)
search = clf.fit(pca_res, y_any_train, sample_weight=weights_any)
f.write('GridSearchCV after PCA')
f.write('Parameters: '+ str(search.best_params_))
f.write('Score: '+ str(search.best_score_))
p = search.best_params_

# Using the best parameters obtained after cross-validation
model_any = xgb.XGBClassifier(scale_pos_weight=0.7102,colsample_bytree=p['colsample_bytree'], 
	learning_rate=p['learning_rate'], max_depth=p['max_depth'], alpha=p['alpha'],
	min_child_weight=p['min_child_weight'],gamma=p['gamma'],n_estimators=p['n_estimators'])

# Training the Model
model_any.fit(pca_res, y_any_train, sample_weight=weights_any)

# Testing the Model on Test Dataset
test_pred_any = model_any.predict(pca_test)

# Testing the Model on Training Dataset
train_pred_any = model_any.predict(pca_res)

# Predicting Probablity of Mutation for Test Dataset
probs_any = model_any.predict_proba(pca_test)

#Training and Testing Accuracy
accuracy_train_any = accuracy_score(y_any_train, train_pred_any)
accuracy_test_any = accuracy_score(y_any_test, test_pred_any)
auc_any = roc_auc_score(y_any_test, test_pred_any)

f.write("Training Accuracy: %.2f%%" % (accuracy_train_any * 100.0))
f.write("Test Accuracy: %.2f%%" % (accuracy_test_any * 100.0))
f.write(str(auc_any))
f.close()