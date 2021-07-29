import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

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

df_y = df['outcome']
df = df.drop(['Sno','outcome','gene', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)

pca = PCA(n_components=2)
pca_res = pca.fit_transform(df)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
lda = LDA()
df_lda = lda.fit(df, df_y).transform(df)
nca = NCA(n_components=2)
df_nca = nca.fit(df, df_y).transform(df)


result_df = pd.DataFrame()
result_df['pca-2d-one'] = pca_res[:,0]
result_df['pca-2d-two'] = pca_res[:,1]

# fig = px.scatter(result_df, x='pca-2d-one', y='pca-2d-two', color=[str(label) for label in y], 
# 	color_discrete_map={'Important Gene Mutation':'#3D6DDD', 'No Mutation':'#FF42B1', 'VUS':'orange'})
# fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)','paper_bgcolor':'rgba(0,0,0,0)', 'width':1200, 'height':680})
# fig.show()

fig = px.scatter(x = df_lda[:,0], y = [0]*len(df_lda), color=[str(label) for label in y], 
	color_discrete_map={'Important Gene Mutation':'#3D6DDD', 'No Mutation':'#FF42B1', 'VUS':'orange'})
fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)','paper_bgcolor':'rgba(0,0,0,0)', 'width':1200, 'height':680})
fig.show()

fig = px.scatter(x = df_nca[:,0], y = df_nca[:,1], color=[str(label) for label in y], 
	color_discrete_map={'Important Gene Mutation':'#3D6DDD', 'No Mutation':'#FF42B1', 'VUS':'orange'})
fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)','paper_bgcolor':'rgba(0,0,0,0)', 'width':1200, 'height':680})
fig.show()