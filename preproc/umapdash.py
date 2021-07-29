import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import umap
from kmodes.kprototypes import KPrototypes
import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
	dbc.Row(
		dbc.Col([
			# html.Div(
			# 	dbc.RadioItems(
			# 		options = [
			# 			{'label':'tsne only', 'value':1},
			# 			{'label':'tsne + pca', 'value':2}
			# 		],
			# 		value = 1,
			# 		id = 'radio',
			# 		className = 'btn-group',
			# 		labelClassName = 'btn btn-secondary',
			# 		labelCheckedClassName = 'active',
			# 	),
			# 	className = 'radio-group',
			# 	style = {'text-align':'center'}
			# ),
			dcc.Graph(id="scatter-plot"),
			html.Div(["KPrototypes n_clusters: ", html.Div(id='display')]),
			dcc.Slider(
				id='n_clusters',
				min=1, max=20, step=1,
				marks={a:str(a) for a in range(21)},
				value=3
			),
			html.Div(["KPrototypes Categorical weight(gamma): ", html.Div(id='display2')]),
			dcc.Slider(
				id='gamma',
				min=0, max=100, step=1,
				marks={a*10:str(a*10) for a in range(11)},
				value=18.306100556757283
			),
			html.Div(["UMAP Categorical weight: ", html.Div(id='display3')]),
			dcc.Slider(
				id='umapcat',
				min=0, max=100, step=1,
				marks={a*10:str(a*10) for a in range(11)},
				value=1
			),
		],
		width=12
		)
	)
])

df = pd.read_csv('../data/model_vus_preprocd.csv')
y=[]
for i in df.index:
	if pd.notna(df.loc[i,'gene']):
		y.append('Important Gene Mutation')
		continue
	if df.loc[i,'outcome'] == 0:
		y.append('No Mutation')
	else:
		y.append('VUS')
df = df.drop(['Sno','outcome','gene','Syndrome', 'gene name if VUS','Ethnicity  0-delhi NCR, west UP, haryana  1-eastern UP, Bihar 2- pahari 3-rajasthan 4-punjab 5-miscellanous'], axis=1)
kdata = df.copy()
cat_columns = [2,3,4,5,6,7,8,9,10,11,12,17,18,25,28,29,30,31,32,33,34,35,40,41,42]
numerical = df.drop(df.columns[cat_columns],axis=1)
categorical = df[df.columns[cat_columns]]
fit1 = umap.UMAP(metric='l2', random_state=1).fit(numerical)
fit2 = umap.UMAP(metric='dice', random_state=1).fit(categorical)


@app.callback(
	Output("scatter-plot", "figure"), 
	[Input("n_clusters", "value"), Input('gamma', 'value'), Input('umapcat', 'value')]
)
def update_bar_chart(n_clusters, gamma, umapcat):
	kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao')
	clusters = kproto.fit_predict(kdata, categorical = cat_columns)
	intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=umapcat)
	intersection = umap.umap_.reset_local_connectivity(intersection)
	embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
													fit1._initial_alpha, fit1._a, fit1._b, 
													fit1.repulsion_strength, fit1.negative_sample_rate, 
													200, 'random', np.random, fit1.metric, 
													fit1._metric_kwds, False, fit1._densmap_kwds, fit1.output_dens)
	# px.scatter(x=embedding[0][:,0], y=embedding[0][:,1], color=[str(x) for x in y], 
	# color_discrete_map={'Important Gene Mutation':'#7C0AFF','No Mutation':'#FF7600','VUS':'#36e1e6'},title = 'UMAP plot').show()
	fig = px.scatter(x=embedding[0][:,0], y=embedding[0][:,1], color=[str(x) for x in clusters], 
		color_discrete_map={'0':'#FF0077','1':'#0A8DFF','2':'#cefd42'},title = 'Clustered plot')
	fig.layout.height=700
	return fig

@app.callback(
	Output('display', 'children'),
	[Input('n_clusters', 'value')]
)
def update_display(val):
	return val

@app.callback(
	Output('display2', 'children'),
	[Input('gamma', 'value')]
)
def update_display2(val):
	return val

@app.callback(
	Output('display3', 'children'),
	[Input('umapcat', 'value')]
)
def update_display3(val):
	return val

app.run_server(debug=True)