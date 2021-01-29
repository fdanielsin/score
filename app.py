from flask import Flask, request, url_for, redirect, render_template
import pickle
import json
import numpy as np
import pandas as pd
import sqlite3
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from config import Config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
print("Model loaded\n")

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def predict_dataset(df):
    
    df_to_pred = df.drop(['SK_ID_CURR'], 1)
    predictions = model.predict(df_to_pred)
    predictions = predictions.reshape((-1,1))
    df['Predictions'] = predictions
    
    return df

def init_dash_file():
    
    html_string = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0"/>
  <title>Home Credit</title>

  <!-- CSS  -->
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <link href="./static/css/materialize.css" type="text/css" rel="stylesheet" media="screen,projection"/>
  <link href="./static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
  
  <!--  Scripts-->
  <script src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
  <script src="./static/js/materialize.js"></script>
  <script src="./static/js/init.js"></script>
</head>

<body>
  <nav class="red" role="navigation">
    <div class="nav-wrapper container"><a id="logo-container" href="https://www.homecredit.net/" class="brand-logo">
        <img src="https://www.homecredit.vn/img/logo-hc2016-main-red.png" width="20%"></a>
      <ul class="right hide-on-med-and-down">
        <li><a href="https://getbootstrap.com/docs/4.3/components/navbar/">Navbar Link</a></li>
      </ul>

      <ul id="nav-mobile" class="sidenav">
        <li><a href="https://getbootstrap.com/docs/4.3/components/navbar/">Navbar Link</a></li>
      </ul>
      <a href="#" data-target="nav-mobile" class="sidenav-trigger"><i class="material-icons">menu</i></a>
    </div>
  </nav>'''
    
    with open("templates/dash.html", "w") as f:
        f.write(html_string)
    
def gen_plots(id_, prediction, features):
    
    value = np.round(prediction[0][1], 2)
    
    # the gauge
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = value,
    mode = "gauge+number",
    title = {'text': "Non-payment probability"},
    gauge = {'axis': {'range': [None, 1]},
             'steps' : [
                 {'range': [0, 0.5], 'color': "lightgreen"},
                 {'range': [0.5, 0.75], 'color': "orange"},
                 {'range': [0.75, 1], 'color': "lightcoral"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.99}}))

    fig.update_layout(
        width=500,
        height=500)

    aPlot = plotly.offline.plot(fig, 
                                config={"displayModeBar": False}, 
                                show_link=False, 
                                include_plotlyjs=False, 
                                output_type='div')
    
    init_dash_file()

    html_string = '''
    <body>
    <div class="container">
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
       <div class="row center">
       <br><br>
      <h5>{{pred}}</h5s>
      ''' + aPlot + '''
      </div>
    </body>
</html>'''
    
    with open("templates/dash.html", "a") as f:
        f.write(html_string)
    
    # the barplot
    
    #features importance
    feat_imp = pd.Series(model.feature_importances_, index=features.drop(['SK_ID_CURR'], axis=1).columns)
    features_imp_df = feat_imp.nlargest(10).to_frame()
    features_imp_df.columns = ['Importance']
    
    fig = px.bar(features_imp_df, x=features_imp_df.index, y="Importance", color="Importance",
            color_continuous_scale=px.colors.sequential.Peach)
    aPlot = plot(fig, 
                                config={"displayModeBar": False}, 
                                show_link=False, 
                                include_plotlyjs=False, 
                                output_type='div')
    html_string = '''
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
       <div class="row center">
      ''' + aPlot + '''
      </div>
'''
    with open("templates/dash.html", "a") as f:
        f.write(html_string)
    
    # the pca
    
    dataset = pd.read_csv("test_data.csv")
    dataset = clean_dataset(dataset)
    dataset = predict_dataset(dataset)
    
    X_for_pca = dataset.drop(['SK_ID_CURR','Predictions'], 1)  
    features_names = features_imp_df.index.tolist()
    scaler = StandardScaler()
    scaler.fit(X_for_pca)
    X_processed = scaler.transform(X_for_pca)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_processed)
    
    dataset["PC1"] = components[:,0]
    dataset["PC2"] = components[:,1]
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(dataset, x="PC1", y="PC2", color="Predictions", color_continuous_scale=px.colors.sequential.Peach)

    for i, feature in enumerate(features_names):
        fig.add_shape(
            type='line',
            x0=2, y0=2,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=10, ay=10,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    
    aPlot = plot(fig, 
                                config={"displayModeBar": False}, 
                                show_link=False, 
                                include_plotlyjs=False, 
                                output_type='div')
    html_string = '''
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
       <div class="row center">
      ''' + aPlot + '''
      </div>
    </div>
 <footer class="page-footer red">
    <div class="container">
      <div class="row">
        <div class="col l6 s12">
          <h5 class="white-text">Our Company</h5>
          <p class="grey-text text-lighten-4">Founded in 1997, Home Credit is an international consumer finance provider with operations in 9 countries. 
              We offer our customers point-of-sale (POS) loans, cash loans and revolving loan products through our online and physical distribution network.</p>


        </div>
        <div class="col l3 s12">
          <h5 class="white-text">More About Us</h5>
          <ul>
            <li><a class="white-text" href="https://www.homecredit.net/about-us/our-history.aspx">Our History</a></li>
            <li><a class="white-text" href="https://www.homecredit.net/about-us/our-products.aspx">Our Products</a></li>
            <li><a class="white-text" href="https://www.homecredit.net/about-us/our-group-structure.aspx">Our Corporate Structure</a></li>
            <li><a class="white-text" href="https://www.homecredit.net/about-us/our-vision-and-business-model.aspx">Our Business Model</a></li>
          </ul>
        </div>
        <div class="col l3 s12">
          <h5 class="white-text">Connect With Us</h5>
          <ul>
            <li><a class="white-text" href="https://customers.homecredit.net/">Our Customers</a></li>
            <li><a class="white-text" href="https://people.homecredit.net/">Our People</a></li>
            <li><a class="white-text" href="https://www.homecredit.net/careers.aspx">Careers Overview</a></li>
            <li><a class="white-text" href="https://www.homecredit.net/careers/recruitment-opportunities.aspx">Recruitement Opportunities</a></li>
          </ul>
        </div>
      </div>
    </div>
    <div class="footer-copyright">
      <div class="container">
      Made by <a class="white-text text-lighten-3" href="http://materializecss.com">Materialize</a>
      </div>
    </div>
  </footer>
  </body>
</html>
'''
    with open("templates/dash.html", "a") as f:
        f.write(html_string)
    


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_request=[int(x) for x in request.form.values()] # takes the requested values from the user as features for the ML model
    #final=[np.array(int_features)] # convert the list of features into an array
    id_ = int_request[0] # The SK ID
    df = pd.read_csv("test_data.csv")
    features_w_id = df[df["SK_ID_CURR"]==id_]
    
    #features without ID column
    features = features_w_id.drop(["SK_ID_CURR"],1)
    prediction=model.predict_proba(features) # predict the probability
    
    gen_plots(id_, prediction, features_w_id) # creates the plots for the dashboard
    
    output='{0:.{1}f}'.format(prediction[0][1], 2) # formats the probability value before printing

    if output>str(0.5):
        return render_template('dash.html', pred='Sorry, your chances to get a loan are low. Your non-payment risk score is high: {}'.format(output))
    else:
        return render_template('dash.html', pred='Your chances to get a loan are good. Your non-payment risk score is low: {}'.format(output))
  

def get_db_connection():
    connection = sqlite3.connect(Config.DATABASE)
    cur = connection.cursor()
    return cur

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == password:
            cur = get_db_connection()
            client = cur.execute(f'SELECT * FROM log_reg WHERE SK_ID_CURR = {username}').fetchone()
            return render_template("loan.html")
    return render_template("login.html")


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000, debug=True)
