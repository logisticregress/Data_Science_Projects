from flask import Flask
from dash.dependencies import Input, Output, State # ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash
import dash_table
import pandas as pd
import snowflake.connector
import copy
import math
import datetime as dt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pmdarima as pmd

df = pd.read_csv('3m_item_data.csv')


df["week"] = pd.to_datetime(df["QUOTE_WEEK"]).dt.week
products = df["ITEM"].unique()
distributors = df['DIST'].unique()

scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

app = dash.Dash(__name__)
# server = app.server

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    )

# @app.route('/')
app.layout = html.Div(
    [
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("trimble01.jpg"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                                },
                            )
                        ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Trimble Analytics Dashboard",
                                    style={"margin-bottom": "0px"},
                                    ),
                                html.H5(
                                    "Quote Event Analysis",
                                    style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("download PDF", id="download-pdf"),
                            # href="https://plot.ly/dash/pricing/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],  # this closes the top div containier, containing the header elements
            id="header",
            className="row flex-display",
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [  # this is the container for the content in the middle of the page
                html.Div(
                    [  # this is the div for the left-nav panel
                        html.P(
                            "Filter by quote date (or select range in histogram):",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="week_slider",
                            min=1,
                            max=52,
                            value=[1, df["week"].max()],
                            className="dcc_control",
                        ),
                        html.P("Select Product:", className="control_label"),
                        dcc.Dropdown(
                            id="product",
                            options=[{'label': i, 'value': i } for i in products],
                            multi=False,
                            #value=list(df["ITEM"].unique()),
                            className="dcc_control",
                        ),
                        html.P("Select Distributor:", className="control_label"),
                        dcc.Dropdown(
                            id="distributor",
                            options=[{'label': i, 'value': i } for i in distributors],
                            multi=False,
                            #value=list(df["ITEM"].unique()),
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="Quotes"), html.P("Quotes")],
                                    id="quotes",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="Contractors"), html.P("Contractors")],
                                    id="cont",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="Distributors"), html.P("Distributors")],
                                    id="dist",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="Estimated Value"), html.P("Estimated Value")],
                                    id="est_val",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="count_graph")],
                                    id="countGraphContainer",
                                    className="pretty_container",
                                ),
                                html.Div(
                                    [dcc.Graph(id="forecast_graph")],
                                    id="forecastGraphContainer",
                                    className="pretty_container",
                                ),
                            ],
                            className="row flex-display",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
                    #className="flex-dispay",
            ),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="geo")],
                        id="geo_loc",
                        className="pretty_container",
                    ),
                    html.Div(
                        [dcc.Graph(id="top_item")],
                        id="topitemcontainer",
                        className="pretty_container",
                    ),
                ],
                className="row flex-display",
            ),
            html.Div(
                [
                html.H5('Manufacturer Data'),
                    dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={
                    'maxHeight': '300',
                    'overflowY': 'scroll'
                    },
                    data=df.to_dict('records'),
                ),
            ],
            className="row flex-display",
            ),

        ],
        id="mainContainer",
        style={"display": "flex", "flex-direction": "column"},
    )

# helper functions
def filter_dataframe(df, product, distributor, week_slider):
    dff = df[
        df["ITEM"].isin(product)
        & df["DIST"].isin(distributor)
        & (df["week"] >= week[0])
        & (df["week"] <= week[1])
    ]
    return dff

# This creates the first graph to show the total quote counts by date
@app.callback(
    Output('count_graph', 'figure'),
    [Input('week_slider', 'value'),
     Input('product', 'value')])
def update_graph(week, prod):
    if prod is None:
        dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
        pv = pd.pivot_table(dff, index=['week'], aggfunc=pd.Series.nunique, fill_value=0)
        pr = pd.DataFrame(pv.reset_index())
        trace1 = go.Scatter(x=pr['week'], y=pr['TX_ID'],
                            line=dict(width=2, color='rgb(229,151,50)'))
        fig = go.Figure(data = [trace1])
        return fig

    else:
        dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
        pv = pd.pivot_table(dff[dff['ITEM'] == prod], index=['week'], aggfunc=pd.Series.nunique, fill_value=0)
        pr = pd.DataFrame(pv.reset_index())
        trace1 = go.Scatter(x=pr['week'], y=pr['TX_ID'],
                            line=dict(width=2, color='rgb(229,151,50)'))
        fig = go.Figure(data = [trace1])
        return fig

@app.callback(
    Output('top_item', 'figure'),
    [Input('week_slider', 'value')])
def update_item_graph(week):
    dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
    pv = pd.pivot_table(dff, values=['TX_ID'], index=['ITEM'], aggfunc=pd.Series.nunique, fill_value=0)
    pr = pd.DataFrame(pv.reset_index())
    pr = pr['TX_ID'].sort_values(ascending=False).head(10)
    trace10 = go.Bar(x=pr['ITEM'],
                    y=pr['TX_ID']
                    )
    fig = go.Figure(data = [trace10])
    return fig



# This creates the first graph to show the total quote counts by date
@app.callback(
    Output('forecast_graph', 'figure'),
    [Input('week_slider', 'value')]
    )
def update_forecast(week):
    pv = pd.pivot_table(df, index=['week'], aggfunc=pd.Series.nunique, fill_value=0)
    pr = pd.DataFrame(pv.reset_index())
    pr = pr[(pr["week"] >= week[0]) & (pr["week"] <= week[1])]

    pr_arima = pmd.auto_arima(pr['TX_ID'], start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=52,
                                 start_P=0, seasonal=False,
                                 d=0, D=0, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    #fc, confint = dfa.predict(n_periods=12, return_conf_int=True)
    fc, conf = pr_arima.predict(n_periods=6, return_conf_int=True)
    fdf = pd.DataFrame(fc)
    # Forecast
    #fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf
    l = len(pr['TX_ID'])
    x_rev = fdf.index[::-1]

    # Make as pandas series
    fc_series = pd.Series(fc, index=fdf.index+l)
    lower_series = pd.Series(conf[:, 0], index=fdf.index+l)
    upper_series = pd.Series(conf[:, 1], index=fdf.index+l)

    # this is the original series
    trace_original = go.Scatter(x=pr.index, y=pr['TX_ID'],
                    line=dict(width=2, color='rgb(255,0,0)'))

    trace_upper = go.Scatter(
            name='Upper 95%',
            x=fdf.index+l,
            y=upper_series,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68,68,68,0.3)',
            fill='tonexty')

    trace_forecast = go.Scatter(
            name='Forecast',
            x=fdf.index+l,
            y=fc_series,
            mode='lines',
            line=dict(width=2, color='rgb(31,119,180)'),
            fillcolor='rgba(68,68,68,0.3)',
            fill='tonexty' )

    trace_lower = go.Scatter(
            name='Lower 95%',
            x=fdf.index+l,
            y=lower_series,
            line=dict(width=0),
            mode='lines'
            )

    layout = go.Layout(
                yaxis=dict(title='Transaction Count'),
                xaxis=dict(title='Week Number'),
                title='Forecast of Transactions',
                showlegend = True)

    fig = go.Figure(data = [trace_lower,trace_forecast,trace_upper, trace_original], layout=layout)
    return fig






# This populates the top set of boxes with summary statistics
@app.callback(
    [
        Output("Quotes", "children"),
        Output("Contractors", "children"),
        Output("Distributors", "children"),
        Output("Estimated Value", "children"),
    ],
    [Input("week_slider", "value")],
    )
def update_text(week):
    #dff = filter_dataframe(df, product, distributor, week_slider)
    dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
    dfff = dff.drop_duplicates(['TX_ID'])
    return '{:,.0f}'.format(dff['TX_ID'].nunique()), dff['CONTRACTOR'].nunique(),dff['DIST'].nunique(), '${:,.2f}'.format(dfff['AVG_TX_PRICE'].sum().round(2))


@app.callback(
        Output('geo', 'figure'),
        [Input('week_slider', 'value'),
         Input('product', 'value')])
def update_geo(week, prod):
    if prod is None:
        dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
        cc = pd.pivot_table(dff, values=['TX_ID'], index=['CONTRACTOR', 'LG', 'LT'], aggfunc=pd.Series.nunique, fill_value=0)
        ccdf = cc.reset_index()
        data = [ go.Scattergeo(
        locationmode = 'USA-states',
        lon = ccdf['LG'],
        lat = ccdf['LT'],
        text = ccdf['CONTRACTOR'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = ccdf['TX_ID'],
            cmax = ccdf['TX_ID'].max(),
            colorbar=dict(
                title="Total Number of Transactions"
            )
        )
        )]

        layout = dict(
        title = 'Total Number of Transactions',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
        )

        fig = go.Figure(data=data, layout=layout )
        return fig

    else:
        dff = df[(df["week"] >= week[0]) & (df["week"] <= week[1])]
        cc = pd.pivot_table(dff[dff['ITEM'] == prod], values=['TX_ID'], index=['CONTRACTOR', 'LG', 'LT'], aggfunc=pd.Series.nunique, fill_value=0)
        ccdf = cc.reset_index()
        data = [ go.Scattergeo(
        locationmode = 'USA-states',
        lon = ccdf['LG'],
        lat = ccdf['LT'],
        text = ccdf['CONTRACTOR'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = ccdf['TX_ID'],
            cmax = ccdf['TX_ID'].max(),
            colorbar=dict(
                title="Total Number of Transactions"
            )
        )
        )]

        layout = dict(
        title = 'Total Number of Transactions',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
        )

        fig = go.Figure(data=data, layout=layout )
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)
