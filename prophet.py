import pandas as pd
import numpy as np
from fbprophet import Prophet
from datetime import datetime, timedelta
import os
import cPickle as pickle

class Model:
	
	
	def __init__(self):
		pass
		
	def train_model(self):
		pass

	def predict(self):
		pass

    
class Mprophet(Model):
	
	def __init__(self):
		self.name = 'mprophet'
	
	def train_model(self, filename):

	    df_raw = pd.read_csv('dataTest/' + filename)
	    df_stock = df_raw.filter(['Date', 'Close'], axis=1)
	    df_stock.columns = ['ds', 'y']

	    model = Prophet()
	    model.fit(df_stock)
	    print("Saving model: {}".format(filename))
	    
	    pickle.dump(model, open('output/models/' + self.name + '/' + filename + ".p", "wb"))


	def predict(self, period, filename):
	    model = pickle.load(open('output/models/' + self.name + '/' + filename + ".p", "rb"))
	    future_data = model.make_future_dataframe(periods = period)
	    forecast = model.predict(future_data)
	    print("Writing to out filename: {}".format(filename))
	    forecast.to_csv('output/prediction/' + self.name + '/' + filename, sep='\t')
	    




