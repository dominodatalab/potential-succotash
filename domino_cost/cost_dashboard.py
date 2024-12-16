import os
import re
import requests
from datetime import datetime, timedelta
from typing import List
import time

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import (
    Dash,
    dash_table,
    dcc,
    html
)
from dash.dependencies import Input, Output, State
from pandas import (
    DataFrame,
    Timestamp
)

api_proxy = os.environ["DOMINO_API_PROXY"]

def get_domino_namespace() -> str:
    api_host = os.environ["DOMINO_API_HOST"]
    pattern = re.compile("(https?://)((.*\.)*)(?P<ns>.*?):(\d*)\/?(.*)")
    match = pattern.match(api_host)
    return match.group("ns")

namespace = get_domino_namespace()

base_url = f"http://domino-cost.{namespace}:9000"
assets_url = f"{base_url}/asset"
allocations_url = f"{base_url}/allocation/summary"

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

NO_TAG = "No tag"

window_to_param = {
    "30d": "Last 30 days",
    "14d": "Last 14 days",
    "7d": "Last 7 days"
}

def get_today_timestamp() -> Timestamp:
    return pd.Timestamp("today", tz="UTC").normalize()

# def get_time_delta(time_span) -> timedelta:
#         if time_span == 'lastweek':
#             days_to_use = 7
#         else:
#             days_to_use = int(time_span[:-1])
#         return timedelta(days=days_to_use-1)

def get_time_delta(time_span: str) -> timedelta:
    try:
        days_to_use = int(time_span[:-1])
        return timedelta(days=days_to_use - 1)
    except (ValueError, TypeError):
        # Default to 30 days if parsing fails
        return timedelta(days=29)

def parse_window(window):
    # Handle common time units like "7d", "24h", etc.
    if window.endswith(('m', 'h', 'd')):
        unit = window[-1]
        value = int(window[:-1])
        now = datetime.utcnow()

        if unit == 'm':
            start_time = now - timedelta(minutes=value)
        elif unit == 'h':
            start_time = now - timedelta(hours=value)
        elif unit == 'd':
            start_time = now - timedelta(days=value)
        return start_time, now

    # Handle relative units like "today", "yesterday", "lastweek", etc.
    elif window in ['today', 'yesterday', 'week', 'lastweek', 'month', 'lastmonth']:
        now = datetime.utcnow()

        if window == 'today':
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        elif window == 'yesterday':
            start_time = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
        elif window == 'week':
            start_time = now - timedelta(days=now.weekday())  # Start of the week
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        elif window == 'lastweek':
            start_time = now - timedelta(days=now.weekday() + 7)  # Start of last week
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=7)
        elif window == 'month':
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        elif window == 'lastmonth':
            start_time = (now.replace(day=1) - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=30)  # Approximate to 30 days
        return start_time, end_time

    # Handle Unix timestamps like "1586822400,1586908800"
    elif ',' in window and all(part.isdigit() for part in window.split(',')):
        start_time, end_time = window.split(',')
        start_time = datetime.utcfromtimestamp(int(start_time))
        end_time = datetime.utcfromtimestamp(int(end_time))
        return start_time, end_time

    # Handle RFC3339 pairs like "2020-04-01T00:00:00Z,2020-04-03T00:00:00Z"
    elif ',' in window and 'T' in window:
        start_time, end_time = window.split(',')
        start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
        end_time = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")
        return start_time, end_time

    raise ValueError(f"Unrecognized window format: {window}")

def filter_allocations_by_time_span(allocations, time_span):
    start_time, end_time = parse_window(time_span)
    
    # Filter allocations by the parsed window
    filtered_allocations = [
        allocation for allocation in allocations 
        if datetime.strptime(allocation['window']['start'], "%Y-%m-%dT%H:%M:%SZ") >= start_time and
           datetime.strptime(allocation['window']['end'], "%Y-%m-%dT%H:%M:%SZ") <= end_time
    ]
    
    return filtered_allocations

def get_aggregated_allocations(selection: str) -> List:
    params = {
        "window": selection,
        "aggregate": (
            "label:dominodatalab.com/workload-type,"
            "label:dominodatalab.com/project-id,"
            "label:dominodatalab.com/project-name,"
            "label:dominodatalab.com/starting-user-username,"
            "label:dominodatalab.com/organization-name,"
            # "label:dominodatalab.com/billing-tag,"
        ),
        "accumulate": False,
        "shareIdle": True,
    }

    res = requests.get(allocations_url, params=params, headers=auth_header)

    res.raise_for_status()
    alloc_data = res.json()["data"]

    #print('alloc_data; ', alloc_data)

    return alloc_data

def get_execution_cost_table(aggregated_allocations: List) -> DataFrame:

    exec_data = []

    storage_cost_keys = ["pvCost", "ramCost", "pvCostAdjustment", "ramCostAdjustment"]

    for costData in aggregated_allocations:

        # workload_type, project_id, project_name, username, organization, billing_tag = costData["name"].split("/")
        # workload_type, project_id, project_name, username, organization = costData["name"].split("/")
        try:
            workload_type, project_id, project_name, username, organization = costData["name"].split("/")
        except Exception:
            if costData["name"] == "__idle__":
                workload_type = project_id = project_name = username = organization = "__idle__"
            else:
                print("Malformed allocation name obtained. Skipping it. Please review.", costData["name"])
        
        #cpu_cost = costData["cpuCost"] + costData["cpuCostAdjustment"]
        cpu_cost = costData.get("cpuCost", 0) + costData.get("cpuCostAdjustment", 0)
        gpu_cost = costData.get("gpuCost", 0) + costData.get("gpuCostAdjustment", 0)

        compute_cost = cpu_cost + gpu_cost
        
        #ram_cost = costData["ramCost"] + costData["ramCostAdjustment"]
        ram_cost = costData.get("ramCost", 0) + costData.get("ramCostAdjustment", 0)
        
        total_cost = costData["totalCost"]

        storage_cost = total_cost - compute_cost
        
        # Change __unallocated__ billing tag into "No Tag"
        # billing_tag = billing_tag if billing_tag != '__unallocated__' else NO_TAG

        exec_data.append({
            "TYPE": workload_type,
            "PROJECT NAME": project_name,
            # "BILLING TAG": billing_tag,
            "USER": username,
            "ORGANIZATION": organization,
            "START": costData["window"]["start"],
            "END": costData["window"]["end"],
            "CPU COST": cpu_cost,
            "GPU COST": gpu_cost,
            "COMPUTE COST": compute_cost,
            "MEMORY COST": ram_cost,
            "STORAGE COST": storage_cost,
            "TOTAL COST": total_cost
        })
    execution_costs = pd.DataFrame(exec_data)

    execution_costs['START'] = pd.to_datetime(execution_costs['START'])
    execution_costs['FORMATTED START'] = execution_costs['START'].dt.strftime('%B %-d')

    return execution_costs

def buildHistogram(cost_table: DataFrame, bin_by: str):
    top = clean_df(cost_table, bin_by).groupby(bin_by)['TOTAL COST'].sum().nlargest(10).index
    costs = cost_table[cost_table[bin_by].isin(top)]
    data_index = costs.groupby(bin_by)['TOTAL COST'].sum().sort_values(ascending=False).index
    title = "Top " + bin_by.title() + " by Total Cost"
    chart = px.histogram(costs, x='TOTAL COST', y=bin_by, orientation='h',
                              title=title, labels={bin_by: bin_by.title(), 'TOTAL COST': 'Total Cost'},
                              hover_data={'TOTAL COST': '$:.2f'},
                              category_orders={bin_by: data_index})
    chart.update_layout(
        title_text=title,
        title_x=0.5,
        xaxis_tickprefix = '$',
        xaxis_tickformat = ',.',
        yaxis={  # Trim labels that are larger than 15 chars
            'tickmode': 'array',
            'tickvals': data_index,
            'ticktext': [f"{txt[:15]}..." if len(txt) > 15 else txt for txt in chart['layout']['yaxis']['categoryarray']]
        },
        dragmode=False
    )
    chart.update_xaxes(title_text="Total Cost")
    chart.update_traces(hovertemplate='$%{x:.2f}<extra></extra>')

    return chart

DOMINO_PROJECT_OWNER = os.environ.get("DOMINO_PROJECT_OWNER")
DOMINO_PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME")
DOMINO_RUN_ID = os.environ.get("DOMINO_RUN_ID")

requests_pathname_prefix = '/{}/{}/r/notebookSession/{}/'.format(DOMINO_PROJECT_OWNER,DOMINO_PROJECT_NAME,DOMINO_RUN_ID)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], routes_pathname_prefix=None, requests_pathname_prefix=requests_pathname_prefix)

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
            html.P("Organization", style={"float": "right", "margin-top": "5px"}),
            width=1
        ),
        dbc.Col(
            dcc.Dropdown(
                id='organization_select',
                options = ['No data'],
                clearable = True,
                searchable = True,
                style={"width": "100%", "whiteSpace":"nowrap"}
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
                searchable = True,
                style={"width": "100%", "whiteSpace":"nowrap"}
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
                searchable = True,
                style={"width": "100%", "whiteSpace":"nowrap"}
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
                id='execution_chart',
                config = {
                    'displayModeBar': False
                }
            )
        )
    ]),
    html.H4('Workload Cost Details', style={"margin-top": "50px"}),
    html.Div(id='table-container'),
    dbc.Row([], style={"margin-top": "50px"}),
    # Add download buttons
    html.Div([
        html.Button("Download All Raw Data CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
        html.Button("Download All Raw Data JSON", id="btn_json"),
        dcc.Download(id="download-dataframe-json")
    ]),
    dbc.Row([], style={"margin-top": "50px"})    
], className="container")

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    # Fetch your execution cost table data here
    aggregated_allocations = get_aggregated_allocations("30d")  # Use the appropriate time window
    cost_table = get_execution_cost_table(aggregated_allocations)
    
    return dcc.send_data_frame(cost_table.to_csv, "execution_costs.csv")

@app.callback(
    Output("download-dataframe-json", "data"),
    Input("btn_json", "n_clicks"),
    prevent_initial_call=True,
)
def download_json(n_clicks):
    # Fetch your execution cost table data here
    aggregated_allocations = get_aggregated_allocations("30d")  # Use the appropriate time window
    cost_table = get_execution_cost_table(aggregated_allocations)
    
    return dcc.send_data_frame(cost_table.to_json, "execution_costs.json", orient="records")
    
def clean_values(values_list: list) -> list:
    """
    remove "__unallocated__" from values'
    """
    return values_list[1:] if values_list[0].startswith("__") else values_list

def clean_df(df: DataFrame, col: str) -> DataFrame:
    """
    remove "__unallocated__" records from dataframe for preview.
    """
    return df[~df[col].str.startswith("__")]

def get_dropdown_filters(cost_table: DataFrame) -> tuple:
    """ 
    First obtain a unique sorted list for each dropdown
    """
    users_list = clean_values(sorted(cost_table['USER'].unique().tolist(), key=str.casefold))
    projects_list = clean_values(sorted(cost_table['PROJECT NAME'].unique().tolist(), key=str.casefold))
    organizations_list = clean_values(sorted(cost_table['ORGANIZATION'].unique().tolist(), key=str.casefold))

    # unique_billing_tags = cost_table['BILLING TAG'].unique()
    # unique_billing_tags = unique_billing_tags[unique_billing_tags != NO_TAG]
    # billing_tags_list = sorted(unique_billing_tags.tolist(), key=str.casefold)
    # billing_tags_list.insert(0, NO_TAG)

    # For each dropdown data return a dict containing the value/label/title (useful for tooltips)
    # billing_tags_dropdown = [{"label": billing_tag, "value": billing_tag, "title": billing_tag} for billing_tag in billing_tags_list]
    projects_dropdown = [{"label": project, "value": project, "title": project} for project in projects_list]
    users_dropdown = [{"label": user, "value": user, "title": user} for user in users_list]
    organizations_dropdown = [{"label": organization, "value": organization, "title": organization} for organization in organizations_list]

    return organizations_dropdown, projects_dropdown, users_dropdown


def get_cost_cards(cost_table: DataFrame) -> tuple[str]:
    total_sum = "${:.2f}".format(cost_table['TOTAL COST'].sum())
    compute_sum = "${:.2f}".format(cost_table['COMPUTE COST'].sum())
    storage_sum = "${:.2f}".format(cost_table['STORAGE COST'].sum())

    return total_sum, compute_sum, storage_sum


def get_cumulative_cost_graph(cost_table: DataFrame, time_span: timedelta):
    x_date_series = pd.date_range(get_today_timestamp() - get_time_delta(time_span), get_today_timestamp()).strftime('%B %-d')
    cost_table_grouped_by_date = cost_table.groupby('FORMATTED START')

    cumulative_cost_graph = {
        'data': [
            go.Bar(
                x=x_date_series,
                y=cost_table_grouped_by_date[column].sum().reindex(x_date_series, fill_value=0),
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
    return cumulative_cost_graph


def get_histogram_charts(cost_table: DataFrame):
    user_chart = buildHistogram(cost_table, 'USER')
    project_chart = buildHistogram(cost_table, 'PROJECT NAME')
    org_chart = buildHistogram(cost_table, 'ORGANIZATION')
    execution_chart = buildHistogram(cost_table, 'TYPE')
    return project_chart, user_chart, org_chart, execution_chart
    # , tag_chart


def workload_cost_details(cost_table: DataFrame):
    formatted = {'locale': {}, 'nully': '', 'prefix': None, 'specifier': '$,.2f'}
    table = dash_table.DataTable(
        columns=[
            {'name': "TYPE", 'id': "TYPE"},
            {'name': "PROJECT NAME", 'id': "PROJECT NAME"},
            {'name': "ORGANIZATION", 'id': "ORGANIZATION"},
            {'name': "USER", 'id': "USER"},
            {'name': "START DATE", 'id': "FORMATTED START"},
            {'name': "CPU COST", 'id': "CPU COST", 'type': 'numeric', 'format': formatted},
            {'name': "GPU COST", 'id': "GPU COST", 'type': 'numeric', 'format': formatted},
            {'name': "STORAGE COST", 'id': "STORAGE COST", 'type': 'numeric', 'format': formatted},
        ],
        data=clean_df(cost_table, "TYPE").to_dict('records'),
        page_size=10,
        sort_action='native',
        style_cell={'fontSize': '11px'},
        style_header={
            'backgroundColor': '#e5ecf6',
            'fontWeight': 'bold'
        },
        export_format="csv",
    )
    return table

@app.callback(
    [
        Output('organization_select', 'options'),
        Output('project_select', 'options'),
        Output('user_select', 'options'),
        Output('totalcard', 'children'),
        Output('computecard', 'children'),
        Output('storagecard', 'children'),
        Output('cumulative-daily-costs', 'figure'),
        Output('project_chart', 'figure'),
        Output('user_chart', 'figure'),        
        Output('org_chart', 'figure'),
        Output('execution_chart', 'figure'),
        Output('table-container', 'children')
    ],
    [
        Input('time_span_select', 'value'),
        Input('organization_select', 'value'),
        Input('project_select', 'value'),
        Input('user_select', 'value')
    ]
)
def update(time_span, organization, project, user):
# def update(time_span, billing_tag, project, user):
    
    allocations = get_aggregated_allocations(time_span)
    if not allocations:
        # return [], [], [], 'No data', 'No data', 'No data', {}, None, None, None, None, None
        return [], [], [], 'No data', 'No data', 'No data', {}, None, None, None, None, None

    allocations = filter_allocations_by_time_span(allocations, time_span)
    
    cost_table = get_execution_cost_table(allocations)

    if user is not None:
        cost_table = cost_table[cost_table['USER'] == user]

    if project is not None:
        cost_table = cost_table[cost_table['PROJECT NAME'] == project]

    if organization is not None:
        cost_table = cost_table[cost_table['ORGANIZATION'] == organization]

    return (
        *get_dropdown_filters(cost_table),
        *get_cost_cards(cost_table),
        get_cumulative_cost_graph(cost_table, time_span),
        *get_histogram_charts(cost_table),
        workload_cost_details(cost_table),
    )


@app.callback(
    [Output('user_select', 'value')],
    [Input('user_chart', 'clickData')]
    # [State('user_select', 'value')]

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
    # [State('project_select', 'value')]

)
def project_clicked(clickData):
    if clickData is not None:
        x_value = clickData['points'][0]['y']
        return [x_value]
    else:
        return [None]

@app.callback(
    [Output('organization_select', 'value')],
    [Input('org_chart', 'clickData')]
    # [State('project_select', 'value')]

)
def org_clicked(clickData):
    if clickData is not None:
        x_value = clickData['points'][0]['y']
        return [x_value]
    else:
        return [None]        

# @app.callback(
#     [Output('billing_select', 'value')],
#     [Input('tag_chart', 'clickData')]
#     [State('billing_select', 'value')]
# )
# def billing_tag_clicked(clickData):
#     if clickData is not None:
#         x_value = clickData['points'][0]['y']
#         return [x_value]
#     else:
#         return [None]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8888) # Domino hosts all apps at 0.0.0.0:8888
