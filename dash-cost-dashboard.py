import subprocess
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
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

# For hitting the API
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

# For interacting with the different scopes
breakdown_options = ["Projects", "User", "Organization"]
breakdown_to_param = {
    "Projects": "dominodatalab.com/project-name",
    "User": "dominodatalab.com/starting-user-username",
    "Organization": "dominodatalab.com/organization-name",
}


# For granular aggregations
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
        cpu_cost = round(sum([costData.get(k,0) for k in cpu_cost_key]), 2)
        gpu_cost = round(sum([costData.get(k,0) for k in gpu_cost_key]), 2)
        compute_cost = round(cpu_cost + gpu_cost, 2)
        storage_cost = round(sum([costData.get(k,0) for k in storage_cost_keys]), 2)
        total_cost = round(compute_cost + storage_cost, 2)
        exec_data.append({
            "TYPE": workload_type,
            "PROJECT NAME": project_name,
            "BILLING TAG": billing_tag,
            "USER": username,
            "START": costData["window"]["start"],
            "END": costData["window"]["end"],
            "CPU COST": f"${cpu_cost}",
            "GPU COST": f"${gpu_cost}",
            "COMPUTE COST": f"${compute_cost}",
            "STORAGE COST": f"${storage_cost}",
            "TOTAL COST": f"${total_cost}"
        })
    execution_costs = pd.DataFrame(exec_data)
    
    return execution_costs

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
                value = "lastweek",
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
                html.H4("No data", id='totalcard')
            ])
        ])),
        dbc.Col(dbc.Card(children=[
            dbc.CardBody([
                html.H3("Compute"),
                html.H4("No data", id='computecard')
            ])
        ])),
        dbc.Col(dbc.Card(children=[
            dbc.CardBody([
                html.H3("Storage"),
                html.H4("No data", id='storagecard')
            ])
        ]))
    ], style={"margin-top": "50px"}),
    dcc.Graph(
        id='cumulative-daily-costs',
        config = {
            'displayModeBar': False
        },
        style={'margin-top': '40px'}
    ),
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
      Output('project_chart', 'figure')],
     [Input('time_span_select', 'value'),
      Input('user_select', 'value'),
      Input('project_select', 'value'),
      Input('billing_select', 'value')]
)
def update(time_span, user, project, billing_tag):
    allocations = get_aggregated_allocations(time_span)
    cost_table = get_execution_cost_table(allocations)
    cost_table['TOTAL COST NUMERIC'] = cost_table['TOTAL COST'].str.replace('[\$,]', '', regex=True).astype(float)
    cost_table['START'] = pd.to_datetime(cost_table['START'])
    cost_table['FORMATTED START'] = cost_table['START'].dt.strftime('%B %-d')
    
    if user is not None:
        cost_table = cost_table[cost_table['USER'] == user]
        
    if project is not None:
        cost_table = cost_table[cost_table['PROJECT NAME'] == project]
        
    if billing_tag is not None:
        cost_table = cost_table[cost_table['BILLING TAG'] == billing_tag]
    
    total_sum = round(cost_table['TOTAL COST'].replace('[\$,]', '', regex=True).astype(float).sum(), 2)
    compute_sum = round(cost_table['COMPUTE COST'].replace('[\$,]', '', regex=True).astype(float).sum(), 2)
    storage_sum = round(cost_table['STORAGE COST'].replace('[\$,]', '', regex=True).astype(float).sum(), 2)
    
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
    
    top_users = cost_table.groupby('USER')['TOTAL COST NUMERIC'].sum().nlargest(10).index
    top_user_costs = cost_table[cost_table['USER'].isin(top_users)]
    user_chart = px.histogram(top_user_costs, x='TOTAL COST NUMERIC', y='USER', orientation='h', 
                              title='Total Cost by User', labels={'USER': 'User', 'TOTAL COST NUMERIC': 'Total Cost'},
                              category_orders={'USER': top_user_costs.groupby('USER')['TOTAL COST NUMERIC'].sum().sort_values(ascending=False).index})
    user_chart.update_layout(title_text='Top Users by Total Cost', title_x=0.5, xaxis_tickprefix = '$', xaxis_tickformat = ',.')
    user_chart.update_xaxes(title_text="Total Cost")
    
    top_projects = cost_table.groupby('PROJECT NAME')['TOTAL COST NUMERIC'].sum().nlargest(10).index
    top_project_costs = cost_table[cost_table['PROJECT NAME'].isin(top_projects)]
    project_chart = px.histogram(top_project_costs, x='TOTAL COST NUMERIC', y='PROJECT NAME', orientation='h', 
                              title='Total Cost by Project', labels={'PROJECT NAME': 'Project', 'TOTAL COST NUMERIC': 'Total Cost'},
                              category_orders={'PROJECT NAME': top_project_costs.groupby('PROJECT NAME')['TOTAL COST NUMERIC'].sum().sort_values(ascending=False).index})
    project_chart.update_layout(title_text='Top Projects by Total Cost', title_x=0.5, xaxis_tickprefix = '$', xaxis_tickformat = ',.')
    project_chart.update_xaxes(title_text="Total Cost")
    
    return cumulative_cost_graph, html.H4(f'${total_sum}'), html.H4(f'${compute_sum}'), html.H4(f'${storage_sum}'), billing_tags, projects, users, user_chart, project_chart

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8888) # Domino hosts all apps at 0.0.0.0:8888