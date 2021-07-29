import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

import plotly.express as px

tab1_content = 'Any Gene Mutation'
tab2_content = 'Important Gene Mutation'
tab3_content = 'VUS Mutation'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

nav = dbc.NavbarSimple(children = 
		dbc.Container([
			dbc.Row([
				dbc.Col(dbc.Button('About', href = '/about')),
				dbc.Col([
					dbc.DropdownMenu(children = [
						dbc.DropdownMenuItem('Home', href = '/'),
						dbc.DropdownMenuItem(divider = True),
						dbc.DropdownMenuItem('About', href = '/about'),
						dbc.DropdownMenuItem(divider = True),
						dbc.DropdownMenuItem(tab1_content),
						dbc.DropdownMenuItem(tab2_content),
						dbc.DropdownMenuItem(tab3_content)
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

maincontent = dbc.Container([
	dbc.Row(
		dbc.Col([
			html.Br(),
			html.H1('Mutation Prediction in Breast Cancer Patients '),
			html.Br(),
			dbc.Tabs(
				[
					dbc.Tab(tab1_content, label='Any Gene Mutation'),
					dbc.Tab(tab2_content, label='Important Gene Mutation'),
					dbc.Tab(tab3_content, label='VUS Mutation')
				],
			),
		],
		width=12
		)
	)
])

app.layout = html.Div([nav, maincontent])

if __name__ == '__main__':
	app.run_server(debug=True)