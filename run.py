from dash import Dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import State, Input, Output
import plotly.offline as pyo # to display plots
import plotly.graph_objects as go

import flask
from flask import Flask, request, url_for, redirect, render_template

from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware

import pickle

import pandas as pd
import numpy as np

def datagen():
    my_sample_data = np.random.random_sample([100,3])
    cat_g = ["good","bad","worst"] 
    sample_Cat = [cat_g[np.random.randint(0,3)] for i in range(100)]
    Base_Data = pd.DataFrame(my_sample_data,columns=["val_1","val_2","val_3"])
    Base_Data["sample_Cat"] = sample_Cat
       
    return(Base_Data)

cat_g = ["good","bad","worst"] 

# create values and labels for the dropdown
options_list = []
for i in cat_g:
    options_list.append({'label': i, 'value': i})

## 5 - create the plots
def fig_generator(sample_data):
    sample_data = sample_data.reset_index(drop=True)
    sample_data.head()
    plot_data =[]

    for i in range(1,4):
        plot_data.append(go.Scatter(x=sample_data.index, y=sample_data['val_'+ str(i)], name = 'val_'+ str(i) ))
    plot_layout = go.Layout(title = " This plot is generated using plotly  ")

    fig = go.Figure( data = plot_data ,layout = plot_layout)

    return(fig.data,fig.layout)

model = pickle.load(open("model.pkl","rb"))
print("Model loaded\n")

server = flask.Flask(__name__)
dash_app = Dash(__name__, server = server, url_base_pathname='/dashboard/')
dash_app.layout = html.Div(children=[html.Div("Welcome to the dashboard",
                                              style= {"color": "white",
                                                      "text-align": "center","background-color": "blue",
                                                      "border-style": "dotted","display":"inline-block","width":"80%"
                                                      
                                                    }),
                       html.Div(dcc.Dropdown(id = "drop_down_1" ,options= options_list , value= 'good'
                                                       ),style= {
                                                      "color": "green",
                                                      "text-align": "center","background-color": "darkorange",
                                                      "border-style": "dotted","display":"inline-block","width":"20%"
                                                      
                                                    }),
                       html.Div(children=[html.P(
                            id="map-title",
                            children = "Forecast and validation for Facility ",
                        ), html.Div(dcc.Graph(id ="plot_area"))
                                                       ],style= {
                                                      "color": "black",
                                                      "text-align": "center","background-color": "yellow",
                                                      "border-style": "dotted","display":"inline-block","width":"75%",
                                                                                                            
                                                    })],style={"width":"100%",'paffing':10})

@dash_app.callback(Output("plot_area", 'figure'),
              
              [Input("drop_down_1", "value")])

def updateplot(input_cat):
    
    df= datagen()
    sample_data = df[df["sample_Cat"] == input_cat ]
    
    trace,layout = fig_generator(sample_data)
    
    return {
        'data': trace,
        'layout':layout
    }


@server.route('/')

@server.route('/hello')
#default function
def home_page():
    return render_template("loan.html")

@server.route('/predict',methods=['POST','GET'])
def predict():
    int_request=[int(x) for x in request.form.values()] # takes the requested values from the user as features for the ML model
    #final=[np.array(int_features)] # convert the list of features into an array
    id_ = int_request[0] # The SK ID
    df = pd.read_csv("test_data.csv")
    features = df[df["SK_ID_CURR"]==id_]
    features = features.drop(["SK_ID_CURR"],1)
    prediction=model.predict_proba(features) # predict the probability
    output='{0:.{1}f}'.format(prediction[0][1], 2) # formats the probability value
    
    #if output>str(0.5):
        #return render_template('dash.html',pred='Sorry, your chances to get a loan are low. Your non-payment risk score is high: {}'.format(output))
    #else:
    #    return render_template('dash.html',pred='Your chances to get a loan are good. Your non-payment risk score is low: {}'.format(output))

@server.route('/dashboard/',methods=['POST','GET'])
def render_dashboard():
    int_request=[int(x) for x in request.form.values()] # takes the requested values from the user
    return flask.redirect('/dash')

app = DispatcherMiddleware(server, {
    '/dash': dash_app.server})

run_simple('0.0.0.0', 8080, app, use_reloader=True, use_debugger=True)