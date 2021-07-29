# import time
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

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
df = df.drop(['Sno','outcome','gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)


df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
fig = px.scatter(df, x='tsne-2d-one', y='tsne-2d-two', color=y, color_discrete_map={'Important Gene Mutation':'#3D6DDD', 'No Mutation':'#FF42B1', 'VUS':'orange'})
fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)','paper_bgcolor':'rgba(0,0,0,0)', 'width':1200, 'height':680})
fig.show()
# plt.show()