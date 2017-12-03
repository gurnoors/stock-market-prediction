from __future__ import absolute_import, division, print_function

import itertools
import datetime
import matplotlib.pylab as plt # TODO
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import sys
import warnings
import time

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    import prep # noqa

def timeit(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        end = time.time()
        print('%s: Ran in %f mins' % (func.__name__, (end - start) / 60))
        return retval
    return wrapped

def test_stationarity(timeseries,
                      maxlag=None, regression=None, autolag=None,
                      window=None, plot=False, verbose=False):
    '''
    Check unit root stationarity of time series.
    
    Null hypothesis: the series is non-stationary.
    If p >= alpha, the series is non-stationary.
    If p < alpha, reject the null hypothesis (has unit root stationarity).
    
    Original source: http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    
    Function: http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html
    
    window argument is only required for plotting rolling functions. Default=4.
    '''
    
    # set defaults (from function page)
    if regression is None:
        regression = 'c'
    
    if verbose:
        print('Running Augmented Dickey-Fuller test with paramters:')
        print('maxlag: {}'.format(maxlag))
        print('regression: {}'.format(regression))
        print('autolag: {}'.format(autolag))
    
    if plot:
        if window is None:
            window = 4
        #Determing rolling statistics
        rolmean = timeseries.rolling(window=window, center=False).mean()
        rolstd = timeseries.rolling(window=window, center=False).std()
        
        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean ({})'.format(window))
        std = plt.plot(rolstd, color='black', label='Rolling Std ({})'.format(window))
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
    
    #Perform Augmented Dickey-Fuller test:
    dftest = smt.adfuller(timeseries, maxlag=maxlag, regression=regression, autolag=autolag)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value',
                                             '#Lags Used',
                                             'Number of Observations Used',
                                            ])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if verbose:
        print('Results of Augmented Dickey-Fuller Test:')
        print(dfoutput)
    return dfoutput


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.
    
    Original source: https://tomaugspurger.github.io/modern-7-timeseries.html
    '''
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax


def model_resid_stats(model_results,
                      het_method='breakvar',
                      norm_method='jarquebera',
                      sercor_method='ljungbox',
                      verbose=True,
                      ):
    '''More information about the statistics under the ARIMA parameters table, tests of standardized residuals:
    
    Test of heteroskedasticity
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity

    Test of normality (Default: Jarque-Bera)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality

    Test of serial correlation (Default: Ljung-Box)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_serial_correlation.html
    '''
    # Re-run the ARIMA model statistical tests, and more. To be used when selecting viable models.
    (het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
    norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[0]
    sercor_stat, sercor_p = model_results.test_serial_correlation(method=sercor_method)[0]
    sercor_stat = sercor_stat[-1] # last number for the largest lag
    sercor_p = sercor_p[-1] # last number for the largest lag

    # Run Durbin-Watson test on the standardized residuals.
    # The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
    # This statistic will always be between 0 and 4. The closer to 0 the statistic,
    # the more evidence for positive serial correlation. The closer to 4,
    # the more evidence for negative serial correlation.
    # Essentially, below 1 or above 3 is bad.
    dw_stat = sm.stats.stattools.durbin_watson(model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

    # check whether roots are outside the unit circle (we want them to be);
    # will be True when AR is not used (i.e., AR order = 0)
    arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
    # will be True when MA is not used (i.e., MA order = 0)
    maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)
    
    if verbose:
        print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(het_method, het_stat, het_p));
        print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(norm_method, norm_stat, norm_p));
        print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(sercor_method, sercor_stat, sercor_p));
        print('\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw_stat))
        print('\nTest for all AR roots outside unit circle (>1): {}'.format(arroots_outside_unit_circle))
        print('\nTest for all MA roots outside unit circle (>1): {}'.format(maroots_outside_unit_circle))
    
    stat = {'het_method': het_method,
            'het_stat': het_stat,
            'het_p': het_p,
            'norm_method': norm_method,
            'norm_stat': norm_stat,
            'norm_p': norm_p,
            'skew': skew,
            'kurtosis': kurtosis,
            'sercor_method': sercor_method,
            'sercor_stat': sercor_stat,
            'sercor_p': sercor_p,
            'dw_stat': dw_stat,
            'arroots_outside_unit_circle': arroots_outside_unit_circle,
            'maroots_outside_unit_circle': maroots_outside_unit_circle,
            }
    return stat


def model_gridsearch(ts, p_min, d_min, q_min, p_max, d_max, q_max, sP_min, sD_min,
					 sQ_min, sP_max, sD_max, sQ_max, trends, s=None, enforce_stationarity=True,
					 enforce_invertibility=True, simple_differencing=False, plot_diagnostics=False,
					 verbose=False, filter_warnings=True):
    '''Run grid search of SARIMAX models and save results.
    '''
    
    cols = ['p', 'd', 'q', 'sP', 'sD', 'sQ', 's', 'trend',
            'enforce_stationarity', 'enforce_invertibility', 'simple_differencing',
            'aic', 'bic',
            'het_p', 'norm_p', 'sercor_p', 'dw_stat',
            'arroots_gt_1', 'maroots_gt_1',
            'datetime_run']

    # Initialize a DataFrame to store the results
    df_results = pd.DataFrame(columns=cols)

    # # Initialize a DataFrame to store the results
    # results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
    #                            columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    mod_num=0
    for trend,p,d,q,sP,sD,sQ in itertools.product(trends,
                                                  range(p_min,p_max+1),
                                                  range(d_min,d_max+1),
                                                  range(q_min,q_max+1),
                                                  range(sP_min,sP_max+1),
                                                  range(sD_min,sD_max+1),
                                                  range(sQ_min,sQ_max+1),
                                                  ):
        # initialize to store results for this parameter set
        this_model = pd.DataFrame(index=[mod_num], columns=cols)

        if p==0 and d==0 and q==0:
            continue

        try:
            model = sm.tsa.SARIMAX(ts,
                                   trend=trend,
                                   order=(p, d, q),
                                   seasonal_order=(sP, sD, sQ, s),
                                   enforce_stationarity=enforce_stationarity,
                                   enforce_invertibility=enforce_invertibility,
                                   simple_differencing=simple_differencing,
                                  )
            
            if filter_warnings is True:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model_results = model.fit(disp=0)
            else:
                model_results = model.fit()

            if verbose:
                print(model_results.summary())

            if plot_diagnostics:
                model_results.plot_diagnostics();

            stat = model_resid_stats(model_results,
                                     verbose=verbose)

            this_model.loc[mod_num, 'p'] = p
            this_model.loc[mod_num, 'd'] = d
            this_model.loc[mod_num, 'q'] = q
            this_model.loc[mod_num, 'sP'] = sP
            this_model.loc[mod_num, 'sD'] = sD
            this_model.loc[mod_num, 'sQ'] = sQ
            this_model.loc[mod_num, 's'] = s
            this_model.loc[mod_num, 'trend'] = trend
            this_model.loc[mod_num, 'enforce_stationarity'] = enforce_stationarity
            this_model.loc[mod_num, 'enforce_invertibility'] = enforce_invertibility
            this_model.loc[mod_num, 'simple_differencing'] = simple_differencing

            this_model.loc[mod_num, 'aic'] = model_results.aic
            this_model.loc[mod_num, 'bic'] = model_results.bic

            # this_model.loc[mod_num, 'het_method'] = stat['het_method']
            # this_model.loc[mod_num, 'het_stat'] = stat['het_stat']
            this_model.loc[mod_num, 'het_p'] = stat['het_p']
            # this_model.loc[mod_num, 'norm_method'] = stat['norm_method']
            # this_model.loc[mod_num, 'norm_stat'] = stat['norm_stat']
            this_model.loc[mod_num, 'norm_p'] = stat['norm_p']
            # this_model.loc[mod_num, 'skew'] = stat['skew']
            # this_model.loc[mod_num, 'kurtosis'] = stat['kurtosis']
            # this_model.loc[mod_num, 'sercor_method'] = stat['sercor_method']
            # this_model.loc[mod_num, 'sercor_stat'] = stat['sercor_stat']
            this_model.loc[mod_num, 'sercor_p'] = stat['sercor_p']
            this_model.loc[mod_num, 'dw_stat'] = stat['dw_stat']
            this_model.loc[mod_num, 'arroots_gt_1'] = stat['arroots_outside_unit_circle']
            this_model.loc[mod_num, 'maroots_gt_1'] = stat['maroots_outside_unit_circle']

            this_model.loc[mod_num, 'datetime_run'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

            df_results = df_results.append(this_model)
            mod_num+=1
        except:
            continue
    return df_results


def setup():
	pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
	np.set_printoptions(precision=5, suppress=True) # numpy

	pd.set_option('display.max_columns', 100)
	pd.set_option('display.max_rows', 100)

	# seaborn plotting style
	sns.set(style='ticks', context='poster')


def readData():
	gs = web.DataReader("GS", data_source='yahoo', start='2006-01-01',
	                    end='2010-01-01')
	# sns.set(style='ticks', context='talk')
	df = gs['Open']
	# tsplot(df, title='test');
	df = df.resample("W").mean()
	# tsplot(df, title='test')
	# test_stationarity(df)
	return df

@timeit
def find_best_SARIMA_paramenters(df):
	# run model grid search
	p_min = 0
	d_min = 0
	q_min = 0
	p_max = 2
	d_max = 1
	q_max = 2

	sP_min = 0
	sD_min = 0
	sQ_min = 0
	sP_max = 1
	sD_max = 1
	sQ_max = 1

	s=52 # weeks in a year

	# trends=['n', 'c']
	trends=['n']

	enforce_stationarity=True
	enforce_invertibility=True
	simple_differencing=False

	plot_diagnostics=False

	verbose=False

	df_results = model_gridsearch(df,
	                              p_min,
	                              d_min,
	                              q_min,
	                              p_max,
	                              d_max,
	                              q_max,
	                              sP_min,
	                              sD_min,
	                              sQ_min,
	                              sP_max,
	                              sD_max,
	                              sQ_max,
	                              trends,
	                              s=s,
	                              enforce_stationarity=enforce_stationarity,
	                              enforce_invertibility=enforce_invertibility,
	                              simple_differencing=simple_differencing,
	                              plot_diagnostics=plot_diagnostics,
	                              verbose=verbose,
	                              )

	df_results.sort_values(by='bic').head(10)
	best_model = df_results.sort_values(by='bic').head(1)

	p = int(best_model['p'])
	d = int(best_model['d'])
	q = int(best_model['q'])
	sP = int(best_model['sP'])
	sD = int(best_model['sD'])
	sQ = int(best_model['sQ'])
	s = int(best_model['s'])
	print('p=%s\nd=%s\nq=%s\nsP=%s\nsD=%s\nsQ=%s\ns=%s' % (p, d, q, sP, sD, sQ, s))
	# df_results.sort_values(by='bic').head(10)
	return (p, d, q, sP, sD, sQ, s)


def temp(df, p, d, q, sP, sD, sQ, s):
	mod_seasonal = smt.SARIMAX(df, order=(p, d, q), seasonal_order=(sP, sD, sQ, s))
	res_seasonal = mod_seasonal.fit()
	pred = res_seasonal.get_prediction(start='2006-01-08')

	x = res_seasonal.forecast(100)
	y = df

	ax = y.plot(label='observed')
	x.plot(ax=ax, label='Model - out sample', alpha=.7)
	pred.predicted_mean.plot(ax=ax, label='Model - in sample', alpha=.7)

	ax.set_ylabel("stock value")
	plt.legend()
	sns.despine()


def main():
	setup()
	df = readData()
	# p, d, q, sP, sD, sQ, s = find_best_SARIMA_paramenters(df)
	p, d, q, sP, sD, sQ, s = 0, 1, 0, 1, 1, 0, 52
	mod_seasonal = smt.SARIMAX(df, order=(p, d, q), seasonal_order=(sP, sD, sQ, s))
	res_seasonal = mod_seasonal.fit()

	forecast = res_seasonal.forecast(100)
	
	pred = res_seasonal.get_prediction(start='2006-01-08')
	pred_ci = pred.conf_int()
	insample_model = pred.predicted_mean
	
	output_df = pd.concat([insample_model , forecast])
	print(df.head())
	print(df.tail())


class Sarima():
    def __init__(self):
        self.name = 'sarima'
    
    def train_model(self, filename):
        df_raw = readData()
        
        # filename = 'BSE_BOM500002.csv'
        df_raw = pd.read_csv('dataTest/' + filename)
        df_stock = df_raw.filter(['Date', 'Close'], axis=1)
        df_stock = df_stock.set_index('Date')
        df = df_stock['Close']

        print("Performing grid search...")
        p, d, q, sP, sD, sQ, s = find_best_SARIMA_paramenters(df)
        print("p, d, q, sP, sD, sQ, s = ", p, d, q, sP, sD, sQ, s)

        # p, d, q, sP, sD, sQ, s = 0, 1, 0, 1, 1, 0, 52
        print("Building model...")
        mod_seasonal = smt.SARIMAX(df, order=(p, d, q), seasonal_order=(sP, sD, sQ, s))
        res_seasonal = mod_seasonal.fit()


        model = res_seasonal
        print("Saving model: {}".format(filename))
        pickle.dump(model, open('output/models/' + self.name + '/' + filename + ".p", "wb"))


    def predict(self, period, filename):
        model = pickle.load(open('output/models/' + self.name + '/' + filename + ".p", "rb"))
        
        period_week = (period / 7) + 1
        forecast = model.forecast(period_week)
        
        print("Writing to out filename: {}".format(filename))
        forecast.to_csv('output/prediction/' + self.name + '/' + filename, sep='\t')



if __name__ == '__main__':
	main()
    # s = Sarima()
    # s.train_model('BSE_BOM500002.csv')
    # s.predict(365, 'test_madhur.txt')

