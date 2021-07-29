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




# VUS PREDICTION
#Dataset - VUS mutation only
title_vus = 'VUS Prediction Model'
df_vus = pd.read_csv('data/model_vus_preprocd.csv')
df_vus = df_vus.set_index('Sno')
df_vus_y = df_vus['outcome']
df_vus_X = df_vus.drop(['outcome','gene','gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
X_vus_train, X_vus_test, y_vus_train, y_vus_test = train_test_split(df_vus_X, df_vus_y, test_size= 0.2373, stratify = df_vus_y, random_state = 0)
weights_vus = []
weight_dict_vus = {1:1,0:0.9504}
for i in y_vus_train:
	weights_vus.append(weight_dict_vus[i])


# Building the Model
model_vus = xgb.XGBClassifier(verbosity=0)

#Crossvalidation
# params = {"objective":"binary:logistic",'colsample_bytree': 0.7,'learning_rate': 0.1,
# 	 'max_depth': 5, 'alpha': 0.5, 'tree_method':'gpu_hist','min_child_weight':2, 'gamma':2,'n_estimators':200}

# xgb_cv = xgb.cv(dtrain=dtrain_vus, params=params, nfold=5,
# 					num_boost_round=50, early_stopping_rounds=10, metrics="auc",stratified=True, 
# 					as_pandas=True, seed=1, verbose_eval=True, show_stdv= True)

distributions = {
		"learning_rate"    : [0.1,0.2,0.3] ,
		"max_depth"        : [3,4,5,6],
		"colsample_bytree" : [0.2,0.5,0.8],
		"n_estimators"     : [150,300,500],
		"alpha"            : [0.2,0.5,0.8,1],
		"gamma"	           : [0,2,4],
		"min_child_weight" : [1,3,5]      
}
scorers = {
			'f1_score':make_scorer(f1_score),
			'accuracy_score': make_scorer(accuracy_score),
			'auc': make_scorer(roc_auc_score)
		  }

rskfold = RepeatedStratifiedKFold(n_splits=5, random_state=1,n_repeats=3)
# clf = RandomizedSearchCV(model_vus,random_state=0, param_distributions=distributions, 
# 			cv=rskfold.split(X_vus_train, y_vus_train), scoring=scorers, n_jobs=-1, refit='auc',verbose=0)
clf = GridSearchCV(model_vus, param_grid=distributions, 
			cv=rskfold.split(X_vus_train, y_vus_train), scoring=scorers, n_jobs=-1, refit='accuracy_score', verbose=1)
search = clf.fit(X_vus_train, y_vus_train, sample_weight=weights_vus)
f = open('CV VUS results.txt', 'a')
f.write('GridSearchCV without feature extraction')
f.write('Refit:Accuracy')
f.write('Parameters: ' + str(search.best_params_))
f.write('Score: ' + str(search.best_score_))
p = search.best_params_

# Using the best parameters obtained after cross-validation
model_vus = xgb.XGBClassifier(colsample_bytree=p['colsample_bytree'], 
	learning_rate=p['learning_rate'], max_depth=p['max_depth'], alpha=p['alpha'],
	min_child_weight=p['min_child_weight'],gamma=p['gamma'],n_estimators=p['n_estimators'])

# Training the Model
model_vus.fit(X_vus_train, y_vus_train, sample_weight=weights_vus)

# Testing the Model on Test Dataset
test_pred_vus = model_vus.predict(X_vus_test)

# Testing the Model on Training Dataset
train_pred_vus = model_vus.predict(X_vus_train)

# Predicting Probablity of Mutation for Test Dataset
probs_vus = model_vus.predict_proba(X_vus_test)

#Training and Testing Accuracy
accuracy_train_vus = accuracy_score(y_vus_train, train_pred_vus)
accuracy_test_vus = accuracy_score(y_vus_test, test_pred_vus)
auc_vus = roc_auc_score(y_vus_test, test_pred_vus)

f.write("Training Accuracy: %.2f%%" % (accuracy_train_vus * 100.0))
f.write("Test Accuracy: %.2f%%" % (accuracy_test_vus * 100.0))
f.write(str(auc_vus))

f.close()
f = open('CV results VUS 2.txt', 'a')


### GridSearch after PCA
# X_vus_train.drop('Sno')
# X_vus_test.drop('Sno')
pca = PCA(n_components=20)
pca_res = pca.fit_transform(X_vus_train)
pca_test = pca.fit_transform(X_vus_test)
distributions = {
		"learning_rate"    : [0.1,0.2,0.3] ,
		"max_depth"        : [3,4,5,6],
		"colsample_bytree" : [0.2,0.5,0.8],
		"n_estimators"     : [150,300,500],
		"alpha"            : [0.2,0.5,0.8,1],
		"gamma"	           : [0,2,4],
		"min_child_weight" : [1,3,5]      
}
model_vus = xgb.XGBClassifier(verbosity=0)
clf = GridSearchCV(model_vus, param_grid=distributions, 
			cv=rskfold.split(pca_res, y_vus_train), scoring=scorers, n_jobs=-1, refit='accuracy_score',verbose=1)
search = clf.fit(pca_res, y_vus_train, sample_weight=weights_vus)
f.write('GridSearchCV after PCA')
f.write('Refit:Accuracy')
f.write('Parameters: '+ str(search.best_params_))
f.write('Score: '+ str(search.best_score_))
p = search.best_params_

# Using the best parameters obtained after cross-validation
model_vus = xgb.XGBClassifier(colsample_bytree=p['colsample_bytree'], 
	learning_rate=p['learning_rate'], max_depth=p['max_depth'], alpha=p['alpha'],
	min_child_weight=p['min_child_weight'],gamma=p['gamma'],n_estimators=p['n_estimators'])

# Training the Model
model_vus.fit(pca_res, y_vus_train, sample_weight=weights_vus)

# Testing the Model on Test Dataset
test_pred_vus = model_vus.predict(pca_test)

# Testing the Model on Training Dataset
train_pred_vus = model_vus.predict(pca_res)

# Predicting Probablity of Mutation for Test Dataset
probs_vus = model_vus.predict_proba(pca_test)

#Training and Testing Accuracy
accuracy_train_vus = accuracy_score(y_vus_train, train_pred_vus)
accuracy_test_vus = accuracy_score(y_vus_test, test_pred_vus)
auc_vus = roc_auc_score(y_vus_test, test_pred_vus)

f.write("Training Accuracy: %.2f%%" % (accuracy_train_vus * 100.0))
f.write("Test Accuracy: %.2f%%" % (accuracy_test_vus * 100.0))
f.write(str(auc_vus))
f.close()