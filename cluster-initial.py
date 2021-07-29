import pandas as pd
import numpy as np
import umap
import umap.plot
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv('cluster_data.csv')
data = data.ffill()
# Data preprocessing

for c in data:
	pt = PowerTransformer()
	data[c] = pt.fit_transform(np.array(data[c]).reshape(-1,1))

# Clustering

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(data)

# categorical_weight = 1 /data.shape[1]

#Embedding numerical & categorical
fit = umap.UMAP(metric='l2').fit_transform(data)

# print(embedding[0][:,0])
print("END")
# print(embedding[0][:,1])
# print(clusters)
# fig, ax = plt.subplots()
px.scatter(fit[:,0], fit[:,1], color=[str(x) for x in (data['clinical stage overall'])], color_discrete_map={'0':'#7C0AFF','1':'#FF7600'},title = 'Actual').show()
# plt.show()
px.scatter(fit[:,0], fit[:,1], color=[str(x) for x in clusters], color_discrete_map={'0':'#FF0077','1':'#0A8DFF'},title = 'Clustering').show()