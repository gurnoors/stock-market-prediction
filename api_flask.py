from flask import jsonify
from flask import Flask
from flask import request
from flask import json
from prophet import Mprophet
import os
import shutil
import csv
import pandas as pd
import datetime
from bin_packing import calculate_best_investment as _calculate_best_investment

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

@app.route('/data', methods=['GET'])
def get_data():
    data_table = {}
    result = {}
    money = request.args.get('money')
    period = request.args.get('period')
    data_table = predict(period)
    result = calculate_best_investment(data_table, money)
     
    return result

def predict(period):
    
    
    models = list()
    
    #if(os.path.exists('output')):
        
    #   shutil.rmtree('output')
    
    #os.makedirs('output/models/mprophet')
    #os.makedirs('output/prediction/mprophet')
    
    model_prophet = Mprophet()
    models.append(model_prophet)
    for model in models:
        for filename in os.listdir('dataTest'):
            print filename    
            if os.path.isfile('dataTest/' + filename):
                print("Path exists: {}".format(filename))
                #model.train_model(filename)
                out_filename = filename
                model_prophet.predict(int(period), out_filename)

    
    
    mdata = {}
    ldata = {}
    adata = {}
    
    data_table = {}
    data_table['mprophet'] = mdata
    data_table['lstm'] = ldata
    data_table['sarima'] = adata


    codes = pd.read_csv('BSE-datasets-codes.csv', sep=',', header = None)
    #print codes
    df1 = codes.to_dict(orient = 'records')
    #print df1

    dict_codes = dict()
    for element in df1:
        dict_codes[element[0]] = element[1]
        
    json_data = {}
    for filename in os.listdir('output/prediction/mprophet'):
        data_modelp = dict()    
        print("Prediction Filename :{}".format(filename))
        
        for filename in os.listdir('output/prediction/mprophet'):
            
            
            df = pd.read_csv('output/prediction/mprophet/' + filename, sep='\t')
            df = df.filter(['ds', 'yhat'], axis=1)
            df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
            now = datetime.datetime.now()
            
            todays_price = str(df.loc[df['ds'] == str(now.date()), 'yhat']).split()[1]
            
            mask = (df['ds'] > str(now.date()))
            
            df = df.loc[mask]
            max_price = str(df.loc[df['yhat'].idxmax()]).split()[4]
            max_price_date = str(df.loc[df['yhat'].idxmax()]).split()[1]
        
            last_row = df.tail(1)
            predicted_price = str(last_row['yhat']).split()[1]

        
            file_code = filename.split('.')[0].split('_')

            company_name = dict_codes[file_code[0] + '/' + file_code[1]] 
            print company_name
    
            data = {'name': company_name, 'today_price': todays_price , 'predicted_price': predicted_price, 'max_price': max_price, 'max_price_date': max_price_date}
        data_modelp[company_name] = data
    # json_data = json.dumps(data_modelp)
    return data_modelp

def calculate_best_investment(data_table, money):
    #result = functon from Gurnoor
    result = _calculate_best_investment(data_table, budget=money)
    return result
    
if __name__ == '__main__':
    app.run(debug=True)
