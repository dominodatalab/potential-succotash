import subprocess
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_bootstrap_components as dbc
import json
import pandas as pd
import numpy as np
import plotly
import os
import re
import requests
import plotly.graph_objs as go
import plotly.express as px
from collections import defaultdict
from typing import Dict, List
from datetime import datetime, timedelta

api_proxy = os.environ["DOMINO_API_PROXY"]

def get_domino_namespace() -> str:
    api_host = os.environ["DOMINO_API_HOST"]
    pattern = re.compile("(https?://)((.*\.)*)(?P<ns>.*?):(\d*)\/?(.*)")
    match = pattern.match(api_host)
    return match.group("ns")

namespace = get_domino_namespace()

base_url = f"http://domino-cost.{namespace}:9000"
assets_url = f"{base_url}/asset"
allocations_url = f"{base_url}/allocation"

auth_url = f"{api_proxy}/account/auth/service/authenticate"

class TokenExpiredException(Exception):
    pass

def get_token() -> str:
    orgs_res = requests.get(auth_url)  
    token = orgs_res.content.decode('utf-8')
    if token == "<ANONYMOUS>":
        raise TokenExpiredException("Your token has expired. Please redeploy your Domino Cost App.")
    return token
 
auth_header = { 
    'X-Authorization': get_token()
}

window_to_param = {
    "30d": "Last 30 days",
    "14d": "Last 14 days",
    "lastweek": "Last week"
}

def get_aggregated_allocations(selection):
    params = {
        "window": selection,
        "aggregate": (
            "label:dominodatalab.com/workload-type,"
            "label:dominodatalab.com/project-id,"
            "label:dominodatalab.com/project-name,"
            "label:dominodatalab.com/starting-user-username,"
            "label:dominodatalab.com/organization-name,"
            "label:dominodatalab.com/billing-tag,"
        ),
        "accumulate": False,
    }

    
    res = requests.get(allocations_url, params=params, headers=auth_header)  
    
    res.raise_for_status() 
    alloc_data = res.json()["data"]
   
    filtered = filter(lambda costData: costData["name"] != "__idle__", alloc_data)

    return list(filtered)

def get_execution_cost_table(aggregated_allocations: List) -> pd.DataFrame:

    exec_data = []

    cpu_cost_key = ["cpuCost", "cpuCostAdjustment"]
    gpu_cost_key = ["gpuCost", "gpuCostAdjustment"]
    storage_cost_keys = ["pvCost", "ramCost", "pvCostAdjustment", "ramCostAdjustment"]

    data = [costData for costData in aggregated_allocations if not costData["name"].startswith("__")]
    
    for costData in data:
        workload_type, project_id, project_name, username, organization, billing_tag = costData["name"].split("/")
        cpu_cost = sum([costData.get(k,0) for k in cpu_cost_key])
        gpu_cost = sum([costData.get(k,0) for k in gpu_cost_key])
        compute_cost = cpu_cost + gpu_cost
        storage_cost = sum([costData.get(k,0) for k in storage_cost_keys])
        total_cost = compute_cost + storage_cost
        exec_data.append({
            "TYPE": workload_type,
            "PROJECT NAME": project_name,
            "BILLING TAG": billing_tag,
            "USER": username,
            "ORGANIZATION": organization,
            "START": costData["window"]["start"],
            "END": costData["window"]["end"],
            "CPU COST": cpu_cost,
            "GPU COST": gpu_cost,
            "COMPUTE COST": compute_cost,
            "STORAGE COST": storage_cost,
            "TOTAL COST": total_cost
        })
    execution_costs = pd.DataFrame(exec_data)
    
    return execution_costs

def buildHistogram(cost_table, bin_by):
    top = cost_table.groupby(bin_by)['TOTAL COST'].sum().nlargest(10).index
    costs = cost_table[cost_table[bin_by].isin(top)]
    title = "Top " + bin_by.title() + " by Total Cost"
    chart = px.histogram(costs, x='TOTAL COST', y=bin_by, orientation='h', 
                              title=title, labels={bin_by: bin_by.title(), 'TOTAL COST': 'Total Cost'},
                              hover_data={'TOTAL COST': '$:.2f'},
                              category_orders={bin_by: costs.groupby(bin_by)['TOTAL COST'].sum().sort_values(ascending=False).index})
    chart.update_layout(title_text=title, title_x=0.5, xaxis_tickprefix = '$', xaxis_tickformat = ',.')
    chart.update_xaxes(title_text="Total Cost")
    chart.update_traces(hovertemplate='$%{x:.2f}<extra></extra>')
    
    return chart

requests_pathname_prefix = '/{}/{}/r/notebookSession/{}/'.format(
    os.environ.get("DOMINO_PROJECT_OWNER"),
    os.environ.get("DOMINO_PROJECT_NAME"),
    os.environ.get("DOMINO_RUN_ID")
)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], routes_pathname_prefix=None, requests_pathname_prefix=requests_pathname_prefix)

app.layout = html.Div([
    html.H2('Domino Cost Management Report', style = {'textAlign': 'center', "margin-top": "30px"}),
    dbc.Row([
        dbc.Col(
            html.H4('Data select', style = {"margin-top": "20px"}),
            width=2
        ),
        dbc.Col(
            html.Hr(style = {"margin-top": "40px"}),
            width=10
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.P("Time Span:", style={"float": "right", "margin-top": "5px"}),
            width=1
        ),
        dbc.Col(
            dcc.Dropdown(
                id='time_span_select',
                options = window_to_param,
                value = "30d",
                clearable = False,
                searchable = False
            ),
            width=2
        ),
        dbc.Col(width=9)
    ], style={"margin-top": "30px"}),
    dbc.Row([
        dbc.Col(
            html.H4('Filter data by', style = {"margin-top": "20px"}),
            width=2
        ),
        dbc.Col(
            html.Hr(style = {"margin-top": "40px"}),
            width=10
        )
    ], style={"margin-top": "50px"}),
    dbc.Row([
        dbc.Col(
            html.P("Billing Tag:", style={"float": "right", "margin-top": "5px"}),
            width=1
        ),
        dbc.Col(
            dcc.Dropdown(
                id='billing_select',
                options = ['No data'],
                clearable = True,
                searchable = True
            ),
            width=3
        ),
        dbc.Col(
            html.P("Project:", style={"float": "right", "margin-top": "5px"}),
            width=1
        ),
        dbc.Col(
            dcc.Dropdown(
                id='project_select',
                options = ['No data'],
                clearable = True,
                searchable = True
            ),
            width=3
        ),
        dbc.Col(
            html.P("User:", style={"float": "right", "margin-top": "5px"}),
            width=1
        ),
        dbc.Col(
            dcc.Dropdown(
                id='user_select',
                options = ['No data'],
                clearable = True,
                searchable = True
            ),
            width=3
        ),
    ], style={"margin-top": "30px"}),
    dbc.Row([
        dbc.Col(dbc.Card(children=[
            dbc.CardBody([
                html.H3("Total"),
                html.H4("Loading", id='totalcard')
            ])
        ])),
        dbc.Col(dbc.Card(children=[
            dbc.CardBody([
                html.H3("Compute"),
                html.H4("Loading", id='computecard')
            ])
        ])),
        dbc.Col(dbc.Card(children=[
            dbc.CardBody([
                html.H3("Storage"),
                html.H4("Loading", id='storagecard')
            ])
        ]))
    ], style={"margin-top": "50px"}),
    dcc.Loading(children=[
        dcc.Graph(
            id='cumulative-daily-costs',
            config = {
                'displayModeBar': False
            },
            style={'margin-top': '40px'}
        )
    ], type='default'),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='user_chart',
                config = {
                    'displayModeBar': False
                }
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='project_chart',
                config = {
                    'displayModeBar': False
                }
            )
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='org_chart',
                config = {
                    'displayModeBar': False
                }
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='tag_chart',
                config = {
                    'displayModeBar': False
                }
            )
        )
    ]),
    html.H4('Workload Cost Details', style={"margin-top": "50px"}),
    html.Div(id='table-container')
], className="container")

@app.callback(
     [Output('cumulative-daily-costs', 'figure'),
      Output('totalcard', 'children'),
      Output('computecard', 'children'),
      Output('storagecard', 'children'),
      Output('billing_select', 'options'),
      Output('project_select', 'options'),
      Output('user_select', 'options'),
      Output('user_chart', 'figure'),
      Output('project_chart', 'figure'),
      Output('org_chart', 'figure'),
      Output('tag_chart', 'figure'),
      Output('table-container', 'children')],
     [Input('time_span_select', 'value'),
      Input('user_select', 'value'),
      Input('project_select', 'value'),
      Input('billing_select', 'value')]
)
def update(time_span, user, project, billing_tag):
    allocations = get_aggregated_allocations(time_span)
    cost_table = get_execution_cost_table(allocations)
    cost_table['START'] = pd.to_datetime(cost_table['START'])
    cost_table['FORMATTED START'] = cost_table['START'].dt.strftime('%B %-d')
    
    if user is not None:
        cost_table = cost_table[cost_table['USER'] == user]
        
    if project is not None:
        cost_table = cost_table[cost_table['PROJECT NAME'] == project]
        
    if billing_tag is not None:
        cost_table = cost_table[cost_table['BILLING TAG'] == billing_tag]
    
    total_sum = "${:.2f}".format(cost_table['TOTAL COST'].sum())
    compute_sum = "${:.2f}".format(cost_table['COMPUTE COST'].sum())
    storage_sum = "${:.2f}".format(cost_table['STORAGE COST'].sum())
    
    users = cost_table['USER'].unique().tolist()
    projects = cost_table['PROJECT NAME'].unique().tolist()
    billing_tags = cost_table['BILLING TAG'].unique().tolist()
    
    cumulative_cost_graph = {
        'data': [
            go.Bar(
                x=cost_table['FORMATTED START'],
                y=cost_table[column],
                name=column
            ) for column in ['CPU COST', 'GPU COST', 'STORAGE COST']
        ],
        'layout': go.Layout(
            title='Daily Costs by Type',
            barmode='stack',
            yaxis_tickprefix = '$',
            yaxis_tickformat = ',.'
        )
    }
    
    user_chart = buildHistogram(cost_table, 'USER')
    project_chart = buildHistogram(cost_table, 'PROJECT NAME')
    org_chart = buildHistogram(cost_table, 'ORGANIZATION')
    tag_chart = buildHistogram(cost_table, 'BILLING TAG')
    
    formatted = {'locale': {}, 'nully': '', 'prefix': None, 'specifier': '$,.2f'}
    table = dt.DataTable(
        columns=[
            {'name': "TYPE", 'id': "TYPE"},
            {'name': "PROJECT NAME", 'id': "PROJECT NAME"},
            {'name': "BILLING TAG", 'id': "BILLING TAG"},
            {'name': "USER", 'id': "USER"},
            {'name': "START DATE", 'id': "FORMATTED START"},
            {'name': "CPU COST", 'id': "CPU COST", 'type': 'numeric', 'format': formatted},
            {'name': "GPU COST", 'id': "GPU COST", 'type': 'numeric', 'format': formatted},
            {'name': "STORAGE COST", 'id': "STORAGE COST", 'type': 'numeric', 'format': formatted},
        ],
        data=cost_table.to_dict('records'),
        page_size=10,
        sort_action='native',
        style_cell={'fontSize': '11px'},
        style_header={
            'backgroundColor': '#e5ecf6',
            'fontWeight': 'bold'
        }
    )
    
    return cumulative_cost_graph, html.H4(total_sum), html.H4(compute_sum), html.H4(storage_sum), billing_tags, projects, users, user_chart, project_chart, org_chart, tag_chart, table

@app.callback(
    [Output('user_select', 'value')],
    [Input('user_chart', 'clickData')]
)
def user_clicked(clickData):
    if clickData is not None:
        x_value = clickData['points'][0]['y']
        return [x_value]
    else:
        return [None]
    
@app.callback(
    [Output('project_select', 'value')],
    [Input('project_chart', 'clickData')]
)
def project_clicked(clickData):
    if clickData is not None:
        x_value = clickData['points'][0]['y']
        return [x_value]
    else:
        return [None]

@app.callback(
    [Output('billing_select', 'value')],
    [Input('tag_chart', 'clickData')]
)
def user_clicked(clickData):
    if clickData is not None:
        x_value = clickData['points'][0]['y']
        return [x_value]
    else:
        return [None]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8888) # Domino hosts all apps at 0.0.0.0:8888