import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.io as pio

# =========================
# FIX WEBGL
# =========================
pio.renderers.default = "svg"

# =========================
# ETL PROCESS
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['ChurnFlag'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.dropna(inplace=True)

df['gender'] = df['gender'].str.lower().str.strip()
df['Contract'] = df['Contract'].str.lower().str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.lower().str.strip()

# =========================
# KPI METRICS
# =========================
churn_rate = df['ChurnFlag'].mean() * 100
avg_tenure = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()
top_contract = df[df['Churn'] == 'Yes']['Contract'].mode()[0]

# =========================
# DASH APP
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Customer Churn Interactive Dashboard"

GRAPH_STYLE = {'height': '420px'}

# =========================
# LAYOUT
# =========================
app.layout = dbc.Container([

    html.H2(
        "Customer Churn Interactive Analysis Dashboard 🔍",
        className="dashboard-title"
    ),

    # ===== KPI CARDS =====
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("Churn Rate", className="kpi-title"),
            html.Div(f"{churn_rate:.1f}%", className="kpi-value")
        ], className="card kpi-card"), width=3),

        dbc.Col(html.Div([
            html.Div("Avg Tenure", className="kpi-title"),
            html.Div(f"{avg_tenure:.0f} months", className="kpi-value")
        ], className="card kpi-card"), width=3),

        dbc.Col(html.Div([
            html.Div("Avg Monthly Charges", className="kpi-title"),
            html.Div(f"${avg_monthly:.2f}", className="kpi-value")
        ], className="card kpi-card"), width=3),

        dbc.Col(html.Div([
            html.Div("High Risk Contract", className="kpi-title"),
            html.Div(top_contract.title(), className="kpi-value")
        ], className="card kpi-card"), width=3),
    ]),

    # ===== TABS =====
    dcc.Tabs(
        id='tabs',
        value='overview',
        className='custom-tabs',
        children=[

            # ===== TAB 1: OVERVIEW =====
            dcc.Tab(label='Overview', value='overview', children=[
                dbc.Row([
                    dbc.Col(html.Div(
                        dcc.Graph(
                            figure=px.pie(
                                df,
                                names='Churn',
                                title="Overall Churn Distribution",
                                color='Churn',
                                color_discrete_map={'Yes': '#bc1313', 'No': '#22c55e'}
                            ),
                            style=GRAPH_STYLE
                        ),
                        className="card"
                    ), width=6),

                    dbc.Col(html.Div(
                        dcc.Graph(
                            figure=px.histogram(
                                df,
                                x='Contract',
                                color='Churn',
                                barmode='group',
                                title="Churn by Contract Type",
                                color_discrete_map={'Yes': "#bc1313", 'No': '#22c55e'}
                            ),
                            style=GRAPH_STYLE
                        ),
                        className="card"
                    ), width=6),
                ])
            ]),

            # ===== TAB 2: EDA =====
            dcc.Tab(label='EDA Explorer', value='eda', children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("X Variable"),
                        dcc.Dropdown(
                            id='eda-x',
                            options=[{'label': c, 'value': c}
                                     for c in ['tenure', 'MonthlyCharges', 'TotalCharges']],
                            value='tenure'
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Y Variable"),
                        dcc.Dropdown(
                            id='eda-y',
                            options=[{'label': c, 'value': c}
                                     for c in ['MonthlyCharges', 'TotalCharges', 'ChurnFlag']],
                            value='MonthlyCharges'
                        )
                    ], width=3),
                ]),
                html.Div(dcc.Graph(id='eda-graph'), className="card")
            ]),

            # ===== TAB 3: REGRESSION =====
            dcc.Tab(label='Linear Regression', value='regression', children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Independent Variable"),
                        dcc.Dropdown(
                            id='reg-x',
                            options=[{'label': c, 'value': c}
                                     for c in ['tenure', 'MonthlyCharges']],
                            value='tenure'
                        )
                    ], width=4),

                    dbc.Col([
                        html.Label("Dependent Variable"),
                        dcc.Dropdown(
                            id='reg-y',
                            options=[{'label': 'TotalCharges', 'value': 'TotalCharges'}],
                            value='TotalCharges'
                        )
                    ], width=4),
                ]),
                html.Div(dcc.Graph(id='reg-graph'), className="card"),
                html.Div(id='reg-summary', className="insight-box")
            ]),

            # ===== TAB 4: CLUSTER =====
            dcc.Tab(label='Customer Segmentation', value='cluster', children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Number of Clusters (k)"),
                        dcc.Slider(
                            id='k-slider',
                            min=2, max=6, step=1, value=3,
                            marks={i: str(i) for i in range(2, 7)}
                        )
                    ], width=6),
                ]),
                html.Div(dcc.Graph(id='cluster-graph'), className="card")
            ]),

            # ===== TAB 5: TIME SERIES =====
            dcc.Tab(label='Time Series', value='timeseries', children=[
                html.Div(dcc.Graph(id='ts-graph'), className="card")
            ]),
        ]
    ),

    html.Div("Customer Churn Analysis Dashboard • Business Intelligence Project",
             className="footer")

], fluid=True)

# =========================
# CALLBACKS
# =========================
@app.callback(
    Output('eda-graph', 'figure'),
    Input('eda-x', 'value'),
    Input('eda-y', 'value')
)
def update_eda(x, y):
    fig = px.scatter(df, x=x, y=y, color='Churn',
                     title=f"{x} vs {y}",
                     color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
    fig.update_layout(template='plotly_white')
    return fig


@app.callback(
    Output('reg-graph', 'figure'),
    Output('reg-summary', 'children'),
    Input('reg-x', 'value'),
    Input('reg-y', 'value')
)
def update_regression(x, y):
    X = df[[x]]
    y_data = df[y]

    model = LinearRegression().fit(X, y_data)
    r2 = model.score(X, y_data)

    fig = px.scatter(df, x=x, y=y, trendline='ols',
                     title="Linear Regression Result")
    fig.update_layout(template='plotly_white')

    summary = html.Div([
        html.P(f"💡 Each additional unit of {x} increases {y} by ${model.coef_[0]:.2f}."),
        html.P(f"📊 The model explains {r2*100:.1f}% of total variation."),
        html.P("⚠️ Strong linear relationship observed between tenure and revenue.")
    ])

    return fig, summary


@app.callback(
    Output('cluster-graph', 'figure'),
    Input('k-slider', 'value')
)
def update_cluster(k):
    X = df[['tenure', 'MonthlyCharges']]
    df_cluster = df.copy()
    df_cluster['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    fig = px.scatter(df_cluster, x='tenure', y='MonthlyCharges',
                     color='Cluster',
                     title=f"Customer Segmentation (k={k})")
    fig.update_layout(template='plotly_white')
    return fig


@app.callback(
    Output('ts-graph', 'figure'),
    Input('tabs', 'value')
)
def update_ts(tab):
    ts = df.groupby('tenure')['ChurnFlag'].mean().reset_index()
    fig = px.line(ts, x='tenure', y='ChurnFlag',
                  title="Churn Rate over Tenure")
    fig.update_layout(template='plotly_white')
    return fig

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run_server(host="0.0.0.0", port=port, debug=False)