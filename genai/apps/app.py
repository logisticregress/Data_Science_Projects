import dash_bootstrap_components as dbc
from dash import Dash
import openai
import os
import dash_uploader as du



APP_TITLE = "Dell AI Products :: Chatbot App"

app = Dash(__name__,
            title=APP_TITLE,
            update_title='Loading...',
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.FLATLY])

du.configure_upload(app, r"C:\Users\Ryan_Clukey\python_dev\chatbot_test\docs", use_upload_id=False)

os.environ["OPENAI_API_KEY"] = 'super_secret_key_that_nobody_knows'
