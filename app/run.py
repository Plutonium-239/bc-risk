import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import plotly.express as px
import contents

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.config.suppress_callback_exceptions = True

app.title = 'BCAMP (Breast Cancer Mutation Prediction)'
nav = dbc.NavbarSimple(children = 
		dbc.Container([
			dbc.Row([
				# dbc.Col(dbc.Button('About', href = '/about')),
				dbc.Col([
					dbc.DropdownMenu(children = [
						dbc.DropdownMenuItem('Home', href = '/'),
						dbc.DropdownMenuItem(divider = True),
						# dbc.DropdownMenuItem('About', href = '/about'),
						# dbc.DropdownMenuItem(divider = True),
						dbc.DropdownMenuItem(contents.any_head, href='/any_gene'),
						dbc.DropdownMenuItem(contents.imp_head, href='/path_gene'),
						dbc.DropdownMenuItem(contents.vus_head, href='/vus')
					],
					label = 'Go to',
					direction = 'left',
					nav = True,
					in_navbar = True
					)
				])
			])
		]),
	color = '#ff84ac',
	brand = 'BCAMP (Breast Cancer Mutation Prediction)',
	brand_href = '/',
	sticky = 'top',
	fluid=True
	)

footer = html.Div(["Made by : ", html.A("Yukti Makhija", href = 'https://github.com/yuktimakhija') ," and ", 
	html.A("Samarth Bhatia", href='https://github.com/Plutonium-239')],
	className="bg-dark text-inverse text-center py-4 footer")

maincontent = dbc.Container([
	dbc.Row(
		dbc.Col([
			html.Br(),
			html.H1('Mutation Prediction in Breast Cancer Patients '),
			html.Br(),
			dbc.Tabs(
				[
					dbc.Tab([contents.any_content(), dbc.Button('Predict Mutation of Any Gene', href='/any_gene')], label='Any Gene Mutation'),
					dbc.Tab([contents.imp_content(), dbc.Button('Predict Mutation of Pathogenic Gene', href='/path_gene')], label='Pathogenic Gene Mutation'),
					dbc.Tab([contents.vus_content(), dbc.Button('Predict Variant of Unknown Significance', href='/vus')], label='VUS Mutation')
				],
			),
		],
		width=12
		)
	)
])

app.layout = html.Div([
	dcc.Location(id = 'url', refresh = False),
	html.Div(nav),
	html.Div(id = 'page-content')
])

@app.callback(
	Output('any_output', 'children'),
	[Input('any_table', 'data')])
def any_table(data):
	if (data == [ {x:0 for x in contents.top_any[:,0]} ]):
		return dash.no_update
	df = pd.DataFrame(data)
	prob = contents.model_any.predict_proba(df)
	pos = "Probability that subject has any mutation: "
	return [html.H3([pos]), html.Br(), html.H1(str(str(round(prob[0,1]*100, 3)) + "%"))]

@app.callback(
	Output('imp_output', 'children'),
	[Input('imp_table', 'data')])
def imp_table(data):	
	if (data == [ {x:0 for x in contents.top_imp[:,0]} ]):
		return dash.no_update
	df = pd.DataFrame(data)
	prob = contents.model_imp.predict_proba(df)
	pos = "Probability that subject has an pathogenic gene mutation: "
	return [html.H3([pos]), html.Br(), html.H1(str(str(round(prob[0,1]*100, 3)) + "%"))]

@app.callback(
	Output('vus_output', 'children'),
	[Input('vus_table', 'data')])
def vus_table(data):
	if (data == [ {x:0 for x in contents.top_vus[:,0]} ]):
		return dash.no_update
	df = pd.DataFrame(data)
	prob = contents.model_vus.predict_proba(df)
	pos = "Probability that subject has a VUS mutation: "
	return [html.H3([pos]), html.Br(), html.H1(str(str(round(prob[0,1]*100, 3)) + "%"))]

# Callback which handles urls
@app.callback(
	Output('page-content', 'children'),	
	[Input('url', 'pathname')])
def diplay_page(pathname):
	if pathname == '/any_gene':
		return dbc.Container(contents.any_predict_page(), fluid=True), footer
	if pathname == '/path_gene':
		return dbc.Container(contents.imp_predict_page(), fluid=True), footer
	if pathname == '/vus':
		return dbc.Container(contents.vus_predict_page(), fluid=True), footer
	return maincontent, footer

if __name__ == '__main__':
	app.run_server(debug=True)