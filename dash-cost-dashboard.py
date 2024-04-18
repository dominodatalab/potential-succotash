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
window_options = ["Last 60 days", "Last 30 days", "Last 14 days", "Last week", "Today"]
window_to_param = {
    "Last 60 days": "60d",
    "Last 30 days": "30d",
    "Last 14 days": "14d",
    "Last week": "lastweek",
    "Today": "today",
}
dropdown_options = []
for key in window_to_param.keys():
    option = {}
    option['label'] = key
    option['value'] = key
    dropdown_options.append(option)
window_choice = window_options[3]

def format_datetime(dt_str: str) -> str:
    datetime_object = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    return datetime_object.strftime("%Y/%m/%d, %H:%M:%S")

def to_date(date_string: str) -> str:
    """Converts minute-level date string to day level

    ex:
       to_date(2023-04-28T15:05:00Z) -> 2023-04-28
    """
    dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%d")


def add_day(date: str, days: int) -> str:
    dt = datetime.strptime(date, "%Y-%m-%d")
    dt_new = dt + timedelta(days=days)
    return dt_new.strftime("%Y-%m-%d")


def get_aggregated_allocations():
    params = {
        "window": window_to_param[window_choice],
        "aggregate": (
            "label:dominodatalab.com/workload-type,"
            "label:dominodatalab.com/project-id,"
            "label:dominodatalab.com/project-name,"
            "label:dominodatalab.com/starting-user-username,"
            "label:dominodatalab.com/organization-name,"
            "label:dominodatalab.com/billing-tag,"
        ),
        "accumulate": True,
    }

    
    res = requests.get(allocations_url, params=params, headers=auth_header)  
    
    res.raise_for_status() 
    alloc_data = res.json()["data"]
   
    filtered = filter(lambda costData: costData["name"] != "__idle__", alloc_data)

    return list(filtered)



def get_top_level_cost() -> Dict[str, float]:
    params = {
        "window": window_to_param[window_choice],
        "accumulate": True,
    }

    res = requests.get(assets_url, params=params, headers=auth_header) 
    
    res.raise_for_status() 
    data = res.json()["data"]
    
    accumulated_data = dict()

    for cost_record in data:
        cost_type = cost_record["type"]
        accumulated_data[cost_type] = accumulated_data.get(cost_type, 0) + cost_record["totalCost"]

    overAllCost = {cost_data: round(accumulated_data[cost_data], 2) for cost_data in accumulated_data}
     
    return overAllCost


def get_daily_cost(selection) -> pd.DataFrame:
    window = window_to_param[selection]
    params = {
        "window": window,
        "aggregate": (
            "label:dominodatalab.com/organization-name"
        ),
    }

    res = requests.get(allocations_url, params=params, headers=auth_header) 
    
    res.raise_for_status() 
    data = res.json()["data"]
    
    # May not have all historical days
    alocs = [day for day in data if day]
    
    # Route returns data non-cumulatively. We make it cumulative by summing over the
    # returned windows (could be days, hours, weeks etc)
    daily_costs = defaultdict(dict)

    cpu_costs = ["cpuCost", "cpuCostAdjustment"]
    gpu_costs = ["gpuCost", "gpuCostAdjustment"]
    storage_costs = ["pvCost", "pvCostAdjustment", "ramCost", "ramCostAdjustment"]

    costs = {"CPU": cpu_costs, "GPU": gpu_costs, "Storage": storage_costs}

    # Gets the overall cost per day
    for aloc in alocs:
        start = aloc["window"]["start"]
        for cost_type, cost_keys in costs.items():
            if cost_type not in daily_costs[start]:
                daily_costs[start][cost_type] = 0.0
            daily_costs[start][cost_type] += sum(aloc.get(cost_key,0) for cost_key in cost_keys)

    # Cumulative sum over the daily costs
    cumulative_daily_costs = pd.DataFrame(daily_costs).T.sort_index()

    
    cumulative_daily_costs["CPU"] = (round(cumulative_daily_costs["CPU"].cumsum(),2) if "CPU" in cumulative_daily_costs else 0)
    cumulative_daily_costs["GPU"] = (round(cumulative_daily_costs["GPU"].cumsum(),2) if "GPU" in cumulative_daily_costs else 0)
    cumulative_daily_costs["Storage"] = (round(cumulative_daily_costs["Storage"].cumsum(),2) if "Storage" in cumulative_daily_costs else 0)


    # Unless we are looking at today granularity, rollup values to the day level
    # (they are returned at the 5min level)
    if window != "today":
        cumulative_daily_costs.index = cumulative_daily_costs.index.map(to_date)
        cumulative_daily_costs = cumulative_daily_costs.groupby(level=0).max()

    return cumulative_daily_costs



def get_execution_cost_table(aggregated_allocations: List) -> pd.DataFrame:

    exec_data = []

    cpu_cost_key = ["cpuCost", "gpuCost"]
    gpu_cost_key = ["cpuCostAdjustment", "gpuCostAdjustment"]
    storage_cost_keys = ["pvCost", "ramCost", "pvCostAdjustment", "ramCostAdjustment"]

    data = [costData for costData in aggregated_allocations if not costData["name"].startswith("__")]
    
    for costData in data:
        workload_type, project_id, project_name, username, organization, billing_tag = costData["name"].split("/")
        cpu_cost = round(sum([costData.get(k,0) for k in cpu_cost_key]), 2)
        gpu_cost = round(sum([costData.get(k,0) for k in gpu_cost_key]), 2)
        compute_cost = round(cpu_cost + gpu_cost, 2)
        storage_cost = round(sum([costData.get(k,0) for k in storage_cost_keys]), 2)
        exec_data.append({
            "TYPE": workload_type,
            "PROJECT NAME": project_name,
            "BILLING TAG": billing_tag,
            "USER": username,
            "START": costData["window"]["start"],
            "END": costData["window"]["end"],
            "CPU COST": f"${cpu_cost}",
            "GPU COST": f"${gpu_cost}",
            "COMPUT COST": f"${compute_cost}",
            "STORAGE COST": f"${storage_cost}",

        })
    execution_costs = pd.DataFrame(exec_data)
    if all(windowKey in execution_costs for windowKey in ("START", "END")):
        execution_costs["START"] = execution_costs["START"].apply(format_datetime)
        execution_costs["END"] = execution_costs["END"].apply(format_datetime)
    
    return execution_costs

df = get_daily_cost(window_choice)
for column in df.columns:
    df[column] = df[column].apply(lambda x: "${0:,.0f}".format(x))


requests_pathname_prefix = '/{}/{}/r/notebookSession/{}/'.format(
    os.environ.get("DOMINO_PROJECT_OWNER"),
    os.environ.get("DOMINO_PROJECT_NAME"),
    os.environ.get("DOMINO_RUN_ID")
)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], routes_pathname_prefix=None, requests_pathname_prefix=requests_pathname_prefix)

app.layout = html.Div([
    html.H2('Domino Cost Management Report', style = {'textAlign': 'center'}),
    html.Div(
        dcc.Dropdown(
            id='dropdown',
            options = dropdown_options,
            value = "Last 14 days",
            clearable = False,
            searchable = False
            ),
        style={'width': '200px', 'font-family': 'Helvetica, sans-serif'}
    ),
    dbc.Row([
        dbc.Col(dbc.Card(id='card1', children=[
            dbc.CardBody([
                html.H3("Total"),
                html.H4(id='totalcard')
            ])
        ])),
        dbc.Col(dbc.Card(id='card2', children=[
            dbc.CardBody([
                html.H3("Compute"),
                html.H4(id='computecard')
            ])
        ])),
        dbc.Col(dbc.Card(id='card3', children=[
            dbc.CardBody([
                html.H3("Storage"),
                html.H4(id='storagecard')
            ])
        ]))
    ], style={"margin-top": "30px"}),
    dcc.Graph(
        id='cumulative-daily-costs',
        config = {
            'displayModeBar': False
        }
    )
], className="container")

@app.callback(
     [Output('cumulative-daily-costs', 'figure'),
      Output('totalcard', 'children'),
      Output('computecard', 'children'),
      Output('storagecard', 'children')],
     [Input('dropdown', 'value')]
)
def update(selected_option):
    updated_df = get_daily_cost(selected_option)
    compute_sum = '{0:.2f}'.format(updated_df.iloc[-1]['CPU'] + updated_df.iloc[-1]['GPU'])
    storage_sum = '{0:.2f}'.format(updated_df.iloc[-1]['Storage'])
    total_sum = '{0:.2f}'.format(float(compute_sum) + float(storage_sum))
    
    for column in updated_df.columns:
        updated_df[column] = updated_df[column].apply(lambda x: "${0:,.0f}".format(x))
    
    figure = {
        'data': [
            go.Bar(
                x=updated_df.index,
                y=updated_df[column],
                name=column
            ) for column in updated_df.columns
        ],
        'layout': go.Layout(
            title='Cumulative Cost',
            barmode='stack',
            yaxis_tickprefix = '$',
            yaxis_tickformat = ',.'
        )
    }
    
    return figure, html.H4(f'${total_sum}'), html.H4(f'${compute_sum}'), html.H4(f'${storage_sum}')

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8888) # Domino hosts all apps at 0.0.0.0:8888