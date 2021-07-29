import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
	dbc.Row(
		dbc.Col([
			html.Div(
				dbc.RadioItems(
					options = [
						{'label':'tsne only', 'value':1},
						{'label':'tsne + pca', 'value':2}
					],
					value = 1,
					id = 'radio',
					className = 'btn-group',
					labelClassName = 'btn btn-secondary',
					labelCheckedClassName = 'active',
				),
				className = 'radio-group',
				style = {'text-align':'center'}
			),
			dcc.Graph(id="scatter-plot"),
			html.Div(["PCA n_components: ", html.Div(id='display3')]),
			dcc.Slider(
				id='pca_comp',
				min=0, max=47, step=1,
				marks={a*10:str(a*10) for a in range(5)},
				value=20
			),
			html.Div(["Perplexity: ", html.Div(id='display')]),
			dcc.Slider(
				id='perplexity',
				min=0, max=100, step=1,
				marks={a*10:str(a*10) for a in range(11)},
				value=1
			),
			html.Div(["No of iteratons: ", html.Div(id='display2')]),
			dcc.Slider(
				id='iters',
				min=250, max=5000, step=50,
				marks={a*250:str(a*250) for a in range(4,21)},
				value=300
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


@app.callback(
	Output("scatter-plot", "figure"), 
	[Input("perplexity", "value"), Input('iters', 'value'), Input('pca_comp', 'value'), Input('radio', 'value')]
)
def update_bar_chart(slider_val, itersval, pcacomps, radio):
	pca_res = df
	if radio == 2:
		pca = PCA(n_components=pcacomps)
		pca_res = pca.fit_transform(df)
	tsne = TSNE(n_components=2, verbose=0, perplexity=slider_val, n_iter=itersval)
	tsne_results = tsne.fit_transform(pca_res)
	df['tsne-2d-one'] = tsne_results[:,0]
	df['tsne-2d-two'] = tsne_results[:,1]
	fig = px.scatter(df, x="tsne-2d-one", y="tsne-2d-two", template='plotly_dark', color=y)
	fig.layout.height=700
	return fig

@app.callback(
	Output('display', 'children'),
	[Input('perplexity', 'value')]
)
def update_display(val):
	return val

@app.callback(
	Output('display2', 'children'),
	[Input('iters', 'value')]
)
def update_display2(val):
	return val

app.run_server(debug=True)