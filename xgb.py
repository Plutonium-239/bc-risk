import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
# import lime


# CONFUSION MATRIX
def confuse(df_target,results, name):
	cmat = confusion_matrix(df_target, results)
	a = []
	for row in cmat:
		a.append(list(row))
	# print(cmat,type(cmat))
	# print(a)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	N = 256
	vals = np.ones((N, 4))
	vals[:, 0] = np.linspace(1, 61/256, N)
	vals[:, 1] = np.linspace(66/256, 109/256, N)
	vals[:, 2] = np.linspace(177/256, 221/256, N)
	plt.rcParams.update({'font.size': 30})
	newcmp = ListedColormap(vals)
	cax = ax.matshow(a,cmap = newcmp)
	for (i, j), z in np.ndenumerate(cmat):
		ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
	fig.colorbar(cax)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	ax.xaxis.label.set_size(22)
	ax.yaxis.label.set_size(22)
	ax.tick_params(axis='both', labelsize=22)
	plt.title(name)
	plt.show()

# EVALUATION PARAMETERS (F1 Score, Recall, Precision)
def eval(target,pred):
	print('F-score:',f1_score(target,pred))
	print('Recall: ',recall_score(target,pred))
	print('Precision: ',precision_score(target,pred))

# AUC-ROC Plot
def aucroc(model, df_test, test_target, auc, name):
	lr_probs = model.predict_proba(df_test)
	lr_probs = lr_probs[:, 1]
	fpr, tpr, thresh = roc_curve(test_target, lr_probs)
	# fpr_i, tpr_i, thresh_i = roc_curve(test_target, test_target)
	# fpr_n, tpr_n, thresh_n = roc_curve(test_target, [0.5]*len(test_target))
	plt.plot(fpr, tpr, color= '#3D6DDD', marker='.', label= 'XGBClassifier (AUC = %.2f)' % (auc))
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(name)
	plt.legend()
	plt.show()

# PRECISION-RECALL Curve
def prcurve(model, df_test, test_target, name):
	lr_probs = model.predict_proba(df_test)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, threshold = precision_recall_curve(test_target, lr_probs)
	prc = auc(lr_recall, lr_precision)
	plt.plot(lr_recall, lr_precision, color= '#3D6DDD', marker='.', label='XGBClassifier (AUC = %.2f)' % (prc))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(name)
	plt.legend()
	plt.show()

# MODEL INTERPRETABLITY USING SHAP
def interpret_shap(model,df):
	# compute the SHAP values for every prediction in the dataset
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(df)
	# Force Plot
	shap.force_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:], show=True)
	# plt.show()
	# Feature Imoortance using mean(SHAP)
	# shap.summary_plot(shap_values, df)
	# Feature Importance Bar Plot using mean(|SHAP|)
	# shap.summary_plot(shap_values,df, plot_type="bar")
	# plt.show()
	# sort the features indexes by their importance in the model
	# (sum of SHAP value magnitudes over the validation dataset)
	# top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
	# make SHAP plots of the three most important features
	# for i in range(5):
	# 	shap.dependence_plot(top_inds[i], shap_values, df)
	# shap_interaction_values = explainer.shap_interaction_values(df)
	f = plt.figure()
	ax1 = f.add_subplot(121)
	shap.summary_plot(shap_values, df, plot_type="bar", show=False)
	ax2 = f.add_subplot(122)
	shap.summary_plot(shap_values, df, show=False)
	ax1.set_xlabel('mean(|SHAP value|)\naverage impact on model output magnitude')
	ax1.xaxis.label.set_size(22)
	ax1.yaxis.label.set_size(22)
	ax1.tick_params(axis='both', labelsize=22)
	ax2.set_xlabel('SHAP value\nimpact on model output')
	ax2.xaxis.label.set_size(22)
	ax2.yaxis.label.set_size(22)
	ax2.tick_params(axis='both', labelsize=22)
	ax2.axes.yaxis.set_visible(False)
	plt.show()
		
#MODEL INTERPRETABILITY USING LIME
# def interpret_lime():


# ANY MUTATION PREDICTION
# Dataset - Any mutation (Imp+VUS)	
# title_any = 'Any Gene Mutation Prediction Model'
# df_any = pd.read_csv('data/model_any_gene_preprocd.csv')
# df_any = df_any.set_index('Sno')
# df_any_y = df_any['outcome']
# df_any_X = df_any.drop(['outcome','gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
# X_any_train, X_any_test, y_any_train, y_any_test = train_test_split(df_any_X, df_any_y, test_size= 0.2373, stratify = df_any_y, random_state = 0)

# # Setting up DMatrices
# dtrain_any = xgb.DMatrix(X_any_train, label=y_any_train)
# weights_any = []
# weight_dict_any = {1:0.71,0:1}
# for i in y_any_train:
# 	weights_any.append(weight_dict_any[i])

# # Building the Model
# # model_any = xgb.XGBClassifier(tree_method='gpu_hist', colsample_bytree=0.5, learning_rate=0.15, max_depth=4,
# # 	alpha=0.8,min_child_weight=2,gamma=3,n_estimators=300)
# model_any = xgb.XGBClassifier(scale_pos_weight=0.7102)

# #Crossvalidation
# # params = {"objective":"binary:logistic",'colsample_bytree': 0.7,'learning_rate': 0.1,
# # 	 'max_depth': 5, 'alpha': 0.5, 'tree_method':'gpu_hist','min_child_weight':2, 'gamma':2,'n_estimators':200}

# # xgb_cv = xgb.cv(dtrain=dtrain_any, params=params, nfold=5,
# # 					num_boost_round=50, early_stopping_rounds=10, metrics="auc",stratified=True, 
# # 					as_pandas=True, seed=1, verbose_eval=True, show_stdv= True)

# # distributions = {
# # 		"learning_rate"    : [0.10, 0.15, 0.20, 0.30] ,
# # 		"max_depth"        : [3,4,5,6],
# # 		"colsample_bytree" : [0.1,0.3,0.5,0.7,0.85,1],
# # 		"n_estimators"     : [100,150,200,250,300,400,500],
# # 		"alpha"            : [0,0.1,0.25,0.5,0.7,1],
# # 		"gamma"	           : [0,1,3,5,7],
# # 		"min_child_weight" : [1,3,5,7,9]      
# # }
# # scorers = {
# # 			'f1_score':make_scorer(f1_score),
# # 			'precision_score': make_scorer(precision_score),
# # 			'recall_score': make_scorer(recall_score),
# # 			'accuracy_score': make_scorer(accuracy_score),
# # 			'auc': make_scorer(roc_auc_score)
# # 		  }

# # rskfold = RepeatedStratifiedKFold(n_splits=5, random_state=1,n_repeats=3)
# # # clf = RandomizedSearchCV(model_any,random_state=0, param_distributions=distributions, 
# # # 			cv=rskfold.split(X_any_train, y_any_train), scoring=scorers, n_jobs=-1, refit='auc',verbose=0)
# # clf = GridSearchCV(model_any, param_grid=distributions, 
# # 			cv=rskfold.split(X_any_train, y_any_train), scoring=scorers, n_jobs=-1, refit='auc',verbose=0)
# # search = clf.fit(X_any_train, y_any_train, sample_weight=weights_any)
# # f = open('CV results.txt', 'w')
# # f.write('Parameters:', search.best_params_)
# # f.write('Score:', search.best_score_)
# # p = search.best_params_
# p = {
# 	# Parameters that we are going to tune.
# 	'max_depth':1,
# 	'min_child_weight': 3,
# 	'eta':.3,
# 	'subsample': 1,
# 	'colsample_bytree': 1,
# 	'alpha': 2,
# 	'gamma': 0,
# 	# Other parameters
# 	'tree_method':'gpu_hist'
# }
# # Using the best parameters obtained after cross-validation
# model_any = xgb.XGBClassifier(**p, scale_pos_weight=0.7102)
# # model_any = xgb.XGBClassifier(scale_pos_weight=0.7102,colsample_bytree=p['colsample_bytree'], 
# # 	learning_rate=p['eta'], max_depth=p['max_depth'], alpha=p['alpha'],subsample=p['subsample'],
# # 	min_child_weight=p['min_child_weight'],gamma=p['gamma'],n_estimators=100, tree_method=p['tree_method'])

# # Training the Model
# model_any.fit(X_any_train, y_any_train)

# # Testing the Model on Test Dataset
# test_pred_any = model_any.predict(X_any_test)

# # Testing the Model on Training Dataset
# train_pred_any = model_any.predict(X_any_train)

# # Predicting Probablity of Mutation for Test Dataset
# probs_any = model_any.predict_proba(X_any_test)

# #Training and Testing Accuracy
# accuracy_train_any = accuracy_score(y_any_train, train_pred_any)
# accuracy_test_any = accuracy_score(y_any_test, test_pred_any)
# auc_any = roc_auc_score(y_any_test, test_pred_any)
# print(title_any)
# print("Training Accuracy: %.2f%%" % (accuracy_train_any * 100.0))
# print("Test Accuracy: %.2f%%" % (accuracy_test_any * 100.0))

# # Evaluating the Model
# eval(y_any_test, test_pred_any)

# # Plotting Confusion Matrix
# confuse(y_any_test, test_pred_any, title_any)

# # Plotting AUC-ROC Curve
# aucroc(model_any, X_any_test, y_any_test, auc_any, title_any)

# # Plotting Precision Recall Curve
# prcurve(model_any, X_any_test, y_any_test, title_any)

# # Interpreting the Model using SHAP
# interpret_shap(model_any,df_any_X)
# print('\n')



# # VUS PREDICTION
# #Dataset - VUS mutation only
# title_vus = 'VUS Prediction Model'
# df_vus = pd.read_csv('data/model_vus_preprocd.csv')
# df_vus = df_vus.set_index('Sno')
# df_vus_y = df_vus['outcome']
# df_vus_X = df_vus.drop(['outcome','gene','gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
# X_vus_train, X_vus_test, y_vus_train, y_vus_test = train_test_split(df_vus_X, df_vus_y, test_size= 0.2373, stratify = df_vus_y, random_state = 0)
# weights_vus = []
# weight_dict_vus = {1:1,0:0.9504}
# for i in y_vus_train:
# 	weights_vus.append(weight_dict_vus[i])

# # Building the Model
# model_vus = xgb.XGBClassifier(tree_method='gpu_hist', colsample_bytree=0.7, learning_rate=0.1, max_depth=5,
# 	alpha=0.5,min_child_weight=2,gamma=2,n_estimators=200)

# # Training the Model
# model_vus.fit(X_vus_train, y_vus_train, sample_weight=weights_vus)

# # Testing the Model on Test Dataset
# test_pred_vus = model_vus.predict(X_vus_test)

# #Testing the Model on Training Dataset
# train_pred_vus = model_vus.predict(X_vus_train)

# #Predicting Probablity of Mutation for Test Dataset
# probs_vus = model_vus.predict_proba(X_vus_test)

# #Training and Testing Accuracy
# accuracy_train_vus = accuracy_score(y_vus_train, train_pred_vus)
# accuracy_test_vus = accuracy_score(y_vus_test, test_pred_vus)
# auc_vus = roc_auc_score(y_vus_test, test_pred_vus)
# print(title_vus)
# print("Training Accuracy: %.2f%%" % (accuracy_train_vus * 100.0))
# print("Test Accuracy: %.2f%%" % (accuracy_test_vus * 100.0))

# # Evaluating the Model
# eval(y_vus_test,test_pred_vus)

# # Plotting Confusion Matrix
# confuse(y_vus_test, test_pred_vus, title_vus)

# #Plotting AUC-ROC Curve
# aucroc(model_vus, X_vus_test, y_vus_test, auc_vus, title_vus)

# # Plotting Precision Recall Curve
# prcurve(model_vus, X_vus_test, y_vus_test, title_vus)

# # Interpreting the Model using SHAP
# interpret_shap(model_vus,df_vus_X)
# print('\n')




#IMPORTANT GENE MUTATION PREDICTION
#Dataset- Imp mutation only
title_imp = 'Important Gene Mutation Prediction Model'
df_imp = pd.read_csv('data/model_imp_gene_preprocd.csv')
df_imp = df_imp.set_index('Sno')
df_imp_y = df_imp['outcome']
df_imp_X = df_imp.drop(['outcome','gene','gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(df_imp_X, df_imp_y, test_size= 0.2373, stratify = df_imp_y, random_state = 0)
weights_imp = []
weight_dict_imp = {0:0.2292,1:1}
for i in y_imp_train:
	weights_imp.append(weight_dict_imp[i])

p = {
	# Parameters that we are going to tune.
	'max_depth':5,
	'min_child_weight': 1,
	'subsample': 1,
	'colsample_bytree': 0.2,
	'eta':0.3,
	'alpha': 0,
	'gamma': 2,
	# Other parameters
	'tree_method':'gpu_hist'
}

# Building the Model
model_imp = xgb.XGBClassifier(**p, scale_pos_weight = 4.364)

# Training the Model
model_imp.fit(X_imp_train, y_imp_train)

# Testing the Model on Test Dataset
test_pred_imp = model_imp.predict(X_imp_test)

#Testing the Model on Training Dataset
train_pred_imp = model_imp.predict(X_imp_train)

#Predicting Probablity of Mutation for Test Dataset
probs_imp = model_imp.predict_proba(X_imp_test)

#Training and Testing Accuracy
accuracy_train_imp = accuracy_score(y_imp_train, train_pred_imp)
accuracy_test_imp = accuracy_score(y_imp_test, test_pred_imp)
auc_imp = roc_auc_score(y_imp_test, test_pred_imp)
print(title_imp)
print("Training Accuracy: %.2f%%" % (accuracy_train_imp * 100.0))
print("Test Accuracy: %.2f%%" % (accuracy_test_imp * 100.0))

# Evaluating the Model
eval(y_imp_test,test_pred_imp)

# Plotting Confusion Matrix
confuse(y_imp_test, test_pred_imp, title_imp)

#Plotting AUC-ROC Curve
aucroc(model_imp, X_imp_test, y_imp_test, auc_imp, title_imp)

# Plotting Precision Recall Curve
prcurve(model_imp, X_imp_test, y_imp_test, title_imp)

# Interpreting the Model using SHAP
interpret_shap(model_imp,df_imp_X)
print('\n')