import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import os


# import components
from components.navbar import render_navbar
from components.input import render_chat_input

import dash_uploader as du

# define layout
uploadWidget = html.Div([
                du.Upload(),
            ], className='uploadBox'),

chatbot_layout = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

filelist = os.listdir(r'C:\Users\Ryan_Clukey\python_dev\chatbot_test\docs')

def render_chatbot():
    return html.Div([
        dcc.Store(id="store-conversation", data=""),
        render_navbar(brand_name="Dell AI Products  |  LLM Prototype with Custom Input"),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P("You can provide your own custom data to the Chatbot algorithm to teach it new things!"),
                        du.Upload(),
                        html.Br(),
                        html.P("Things I know about:"),
                        html.Div(filelist),
                    ], width=2),
                    dbc.Col([
                        dbc.Card([
                            chatbot_layout,
                                html.Div(render_chat_input(), style={'margin-left': '70px', 'margin-right': '70px', 'margin-bottom': '20px'}),
                                dbc.Spinner(html.Div(id="loading-component")),

                        ], outline=True),
                        
                    ], width = 8),
                ]),
            ]),
        ),
    ])

