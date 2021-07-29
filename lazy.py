#%%
from lazypredict.Supervised import LazyClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('data/model_any_gene_preprocd.csv')
# df = pd.read_csv('data/model_any_gene_preprocd.csv')
# df = pd.read_csv('data/model1_preprocd.csv')
df = df.set_index('Sno')
df_y = df['outcome']
df_X = df.drop(['outcome','gene','gene name if VUS'], axis=1)
# df_X.columns
#%%
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size= 0.25)
#%%
clf = LazyClassifier(custom_metric=None)
#%%
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
#%%
print(models)
# print(predictions)
# %%
