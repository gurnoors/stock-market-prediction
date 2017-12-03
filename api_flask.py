from flask import jsonify
from flask import Flask
from flask import request
from flask import json
from prophet import Mprophet
import os
import shutil
import csv
import pandas as pd


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
    result = calculate_best_investment(data_table)
     
    return result

def predict(period):
    
	
	models = list()
	
	if(os.path.exists('output')):
		
		shutil.rmtree('output')
	
	os.makedirs('output/models/mprophet')
	os.makedirs('output/prediction/mprophet')
	
	model_prophet = Mprophet()
	models.append(model_prophet)
	for model in models:
		for filename in os.listdir('dataTest'):
		    print filename    
		    if os.path.isfile('dataTest/' + filename):
			print("Path exists: {}".format(filename))
			model.train_model(filename)
			out_filename = filename
			model_prophet.predict(int(period), out_filename)

	#return "Predicted successfully"

	data_table = {}
	"""	
	mdata = {}
	ldata = {}
	adata = {}
	
	data_table = {}
	data_table['mprophet'] = mdata
	data_table['lstm'] = ldata
	data_table['sarima'] = adata

	for filename in os.listdir('output/prediction/mprophet'):
		
		print("Prediction Filename :{}".format(filename))
		
		df = pd.read_csv('output/prediction/mprophet/' + filename)
		df = df.filter(['ds', 'yhat'], axis=1)
		print df	
	
		data_table['df'] = df
	"""
	return data_table

def calculate_best_investment(data_table):
	#result = functon from Gurnoor
	result = {'res': 'true'}
	return jsonify(result)
    
if __name__ == '__main__':
	app.run(debug=True)
