import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import numpy as np
import plotly.express as px
import pickle

any_head = 'Any Gene Mutation'
imp_head = 'Pathogenic Gene Mutation'
vus_head = 'VUS Mutation'

model_any = pickle.load(open('models/any_reduced.pickle', 'rb'))
parameters_any = model_any.get_booster().feature_names
importances_any = list(model_any.feature_importances_)
# top_any = np.array([[a,b] for a,b in zip(parameters_any,importances_any)])

model_imp = pickle.load(open('models/imp_reduced.pickle', 'rb'))
parameters_imp = model_imp.get_booster().feature_names
importances_imp = list(model_imp.feature_importances_)
# top_imp = np.array([[a,b] for a,b in zip(parameters_imp,importances_imp)])

model_vus = pickle.load(open('models/vus_reduced.pickle', 'rb'))
parameters_vus = model_vus.get_booster().feature_names
importances_vus = list(model_vus.feature_importances_)
# top_vus = np.array([[a,b] for a,b in zip(parameters_vus,importances_vus)])

top_any  = np.array([
	['significant family history', 0.03419184684753418],
	['IES-R Score', 0.04934421926736832],
	['tubal ligation', 0.07356269657611847],
	['height', 0.07469271868467331],
	['number of second degree relatives', 0.10489383339881897],
	['weight', 0.1303674876689911],
	['number of NCCN criteria', 0.15290623903274536],
	['breastfeeding duration', 0.16892674565315247],
	['age', 0.1947636753320694],
	['HBOC Syndrome', 0.4236903488636017]
])

top_imp = np.array([
	['family cancer history', 0.096233],
	['N stage', 0.099010],
	['age at first childbirth', 0.107066],
	['family testing done', 0.111328],
	['number of second degree relatives', 0.112008],
	['weight', 0.137927],
	['RRSO advised', 0.224071],
	['FinalDass21Score', 0.319688],
	['RRM advised', 0.696016],
	['HBOC Syndrome', 1.734899]
])

top_vus = np.array([
	['HER2', 0.09351715],
	['tubal ligation', 0.10000973],
	['male children', 0.10008002],
	['age', 0.14124352],
	['number of second degree relatives', 0.14773075],
	['IES-R Score', 0.17928961],
	['breastfeeding duration', 0.23812924]
])

def top_features(top_list):
	names = top_list[:,0]
	vals = np.array(top_list[:,1], dtype=np.float32)
	barplot = px.bar(x = vals, y = names, color = range(len(names)), orientation='h', template='plotly_dark')
	barplot.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
	barplot.update_traces(hovertemplate=("%{y} :  %{x}"))
	return dbc.Container(
		dbc.Row(
			dbc.Col([
				html.H3('Features ranked according to their importances'),
				dcc.Graph(figure = barplot)
			])
		)
	)

def any_content(): return top_features(top_any)
def imp_content(): return top_features(top_imp)
def vus_content(): return top_features(top_vus)
note = '''
The table is editable, you can select cells and type desired values. For a prediction to be made, just click anywhere outside the table.
You can even paste from an excel sheet / CSV file. Example of a formatted CSV file: 
'''
def predict_page(top_list, list_name, head):
	cols = [ {'id':x, 'name':x } for x in top_list[:,0][::-1] ]
	sampledata = [ {x:0 for x in top_list[:,0]} ]
	table = dash_table.DataTable(
		id = list_name+'_table',
		# className = 'table',
		columns = cols,
		data = sampledata,
		editable = True,
		style_header={'backgroundColor': 'rgb(80, 80, 80)'},
		style_cell={
			'backgroundColor': 'rgb(50, 50, 50)',
			'color': 'white',
			'border': '1px solid #ffaaca'
		},
		style_data={
		},
		style_data_conditional=[{
			'if':{
				'state': 'active'
			},
			'backgroundColor': 'rgba(255, 65, 54, 0.2)',
			'color': 'white',
			'border': '2px dashed hotpink'
		}]
	)
	return [dbc.Row(
		dbc.Col([
			html.Br(),
			html.H1('Prediction using the ' + head + ' model'),
			html.Div(table, className='table-outer'),
		])
	),
	dbc.Row(
		dbc.Col([
			dbc.Card(
				html.Div(html.H3('Start inputting some values...'), id = list_name+'_output'),
			),
			html.Br()
		], width=8),
	className='justify-content-center'
	),
	dbc.Row(
		dbc.Col([
			html.Div(note, style={"white-space": "pre-line"}),
			html.A('Download example'),
			dcc.Download(id = 'download-csv')
		], width=8),
	className='justify-content-center'
	)]


def any_predict_page(): return predict_page(top_any, 'any', any_head)
def imp_predict_page(): return predict_page(top_imp, 'imp', imp_head)
def vus_predict_page(): return predict_page(top_vus, 'vus', vus_head)