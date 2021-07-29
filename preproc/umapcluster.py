import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
# import umap
import umap
from kmodes.kprototypes import KPrototypes
import numpy as np


df = pd.read_csv('../data/model_vus_preprocd.csv')
# df = df.set_index('Sno')
# y = np.array(df['outcome'])
y=[]
for i in df.index:
	if pd.notna(df.loc[i,'gene']):
		y.append('Important Gene Mutation')
		continue
	if df.loc[i,'outcome'] == 0:
		y.append('No Mutation')
	else:
		y.append('VUS')
outcome = df['outcome']
df = df.drop(['Sno','outcome', 'gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)

kdata = df.copy()

cat_columns = [2,3,4,5,6,7,8,9,10,11,12,17,18,25,28,29,30,31,32,33,34,35,40,41,42]

kproto = KPrototypes(n_clusters=3, init='Cao')
clusters = kproto.fit_predict(kdata, categorical = cat_columns)

categorical_weight = 1 /df.shape[1]

numerical = df.drop(df.columns[cat_columns],axis=1)
categorical = df[df.columns[cat_columns]]

print('preproc done')
fit1 = umap.UMAP(metric='l2',random_state=1).fit(numerical)
print('numerical clustering done')
# px.scatter(fit1, color=[str(x) for x in (data['Outcome'])], color_discrete_map={'0':'#7C0AFF','1':'#FF7600'}).show()
# px.scatter(fit1, color=[str(x) for x in clusters], color_discrete_map={'0':'#FF0077','1':'#0A8DFF'}).show()
fit2 = umap.UMAP(metric='dice',random_state=1).fit(categorical)
print('categorical clustering done')

categorical_weight = 5
# for categorical_weight in [0.05,]
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
                                                fit1._initial_alpha, fit1._a, fit1._b, 
                                                fit1.repulsion_strength, fit1.negative_sample_rate, 
                                                200, 'random', np.random, fit1.metric, 
                                                fit1._metric_kwds, False, fit1._densmap_kwds, fit1.output_dens)

# print(embedding[0][:,0])
print("END")
# print(embedding[0][:,1])
# print(clusters)
# fig, ax = plt.subplots()
px.scatter(x=embedding[0][:,0], y=embedding[0][:,1], color=[str(x) for x in y], 
	color_discrete_map={'Important Gene Mutation':'#7C0AFF','No Mutation':'#FF7600','VUS':'#36e1e6'},title = 'UMAP plot').show()
px.scatter(x=embedding[0][:,0], y=embedding[0][:,1], color=[str(x) for x in clusters], 
	color_discrete_map={'0':'#FF0077','1':'#0A8DFF','2':'#cefd42'},title = 'Clustered plot').show()