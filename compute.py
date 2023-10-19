# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 02:07:44 2023

@author: kanld
"""

import pandas as pd
import scipy.stats as sstat
import numpy as np
import scipy.optimize as sopt
import arch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def gbm_process(spot, drift, vol, n_days, n_paths = 1):
	
	dt = 1/365   
	growth_path = np.exp((drift - vol **2 /2) * dt + vol * np.random.normal(0, np.sqrt(dt), size = (n_days, n_paths)))
	growth_path = np.cumprod(growth_path, axis = 0)
	price_path = spot * growth_path
		
	return price_path

def mc_garch(daily_return, n_days, start_date = 0, n_paths = 10000):

	start_date = pd.to_datetime(start_date)
	
	model = arch.arch_model(100 * daily_return, vol = 'garch', mean = 'zero')
	fit = model.fit(first_obs = start_date, disp = "off")

	omega = fit.params.loc['omega']
	alpha = fit.params.loc['alpha[1]']
	beta = fit.params.loc['beta[1]']
	lt_mean = np.sqrt(252 * omega / (1 - alpha - beta) / 10000)

	ret = np.zeros((n_days+1, n_paths))
	ret[0] = daily_return[-1]

	var_pred = np.zeros((n_days+1, n_paths))
	var_pred[0] = (fit.conditional_volatility[-1] ** 2) / 10000

	for i in range(n_days):

		pred = omega / 10000 + alpha * ret[i] ** 2 + beta * var_pred[i]
		var_pred[i+1] = pred
		ret_pred = np.random.normal(0, np.sqrt(var_pred[i]))
		ret[i+1] = ret_pred

	var_pred = np.sqrt(252 * var_pred)
	
	return var_pred[n_days], {'params' : fit.params, 'lt_mean' : lt_mean}


def custom_cdf_function(daily_close_vol, target_x, start_x = None):

	'''평균 분산 등과 같은 parameter 없이 data 그 자체에서 pdf 도출'''

	pdf = sstat.gaussian_kde(daily_close_vol)
	
	if start_x == None:
		compute_p = pdf.integrate_box(-np.inf, target_x)
	else:
		compute_p = pdf.integrate_box(start_x, target_x)
		
	return compute_p

def discrete_kelly(p_vector, outcome_vector):
	
	p_vector = np.array(p_vector)
	outcome_vector = np.array(outcome_vector)
	
	# sum of log expected outcomes
	
	def inverted_log_return(kelly_ratio):	
		result = - np.sum(p_vector * np.log(1 + kelly_ratio * outcome_vector))	
		return result

	ratio = sopt.minimize(inverted_log_return, x0 = 0.5).x
	expected_return = -sopt.minimize(inverted_log_return, x0 = 0.5).fun
	
	return ratio, expected_return

def continuous_kelly(mean, stdev):
	
	# 2 order taylor approximation
	
	symmetrical_dist = mean / (mean ** 2 + stdev ** 2)
	
	# 3 order
	
	skewness = 0 ## compute scistat.skew(dataset) from a given sample
	lambda_3 = stdev**3 * skewness + 3 * mean * stdev **2 - mean ** 3
	asymmetrical_dist = symmetrical_dist + (lambda_3 * mean ** 2) / ((mean **2 + stdev **2) **2)
	
	return symmetrical_dist, asymmetrical_dist

def fraction_of_kelly(kelly_ratio, mean, stdev):
	
	growth_rate = (kelly_ratio - np.power(kelly_ratio, 2) / 2) * (mean ** 2 / stdev ** 2)
	
	return growth_rate


class ohlc_return:

    ''' feed any price dataset with OHLC format as df_price'''

    def __init__(self, df_price):

        self.df_price = df_price
    
    def total_return(self, n_days):
            
        '''close : close_over_close / high : high_over_close / low : low_over_close / minmax : high - low / tr : true return (encompassing minmax + more)'''
        
        high_high = self.df_price.rolling(window = n_days)['high'].max()
        low_low = self.df_price.rolling(window = n_days)['low'].min()
        last_close = self.df_price['close'].shift(n_days)

        c_over_c = self.df_price['close'] / last_close - 1 # 1) 최종 종가대비 기준 종가
        h_over_c = high_high / last_close - 1 # 2) 기간 내 고가대비 기준 종가
        l_over_c = low_low / last_close - 1  # 3) 기간 내 저가대비 기준 종가
        h_over_l = (high_high - low_low) / last_close  # 기간 내 고가-저가대비 기준 종가
        tr = pd.concat([np.abs(h_over_c), 
                        np.abs(l_over_c), 
                        h_over_l], axis = 1).max(axis = 1)  
                        # 4) True return = max(h_over_c/l_over_c/h_over_l) 의 절대값 -> 제일 확대
        
        result = pd.concat([c_over_c, h_over_c, l_over_c, tr], axis = 1)
        result.columns = ['close', 'high', 'low', 'tr']

        return result.dropna()
    
    def plus_return(self, n_days):
            
        plus_only = self.total_return(n_days)
        plus_only = plus_only[plus_only > 0].fillna(0)

        return plus_only[['close', 'high']].dropna()
    
    def minus_return(self, n_days):
        
        minus_only = self.total_return(n_days)
        minus_only = minus_only[minus_only < 0].fillna(0)

        return minus_only[['close', 'low']].dropna()

    def intraday_ret(self):
        
        '''h_over_o / l_over_o / c_over_o / h_over_l'''
        
        h_over_o = self.df_price['high'] / self.df_price['open'] - 1 # 시초대비 고가
        l_over_o = self.df_price['low'] / self.df_price['open'] - 1 # 시초대비 저가 
        c_over_o = self.df_price['close'] / self.df_price['open'] - 1 # 시초대비 종가
        h_over_l = self.df_price['high'] / self.df_price['low'] - 1  # 당일고저차

        result = pd.concat([h_over_o, l_over_o, c_over_o, h_over_l], axis = 1)

        return result.dropna()



def volscore(df_return, price = 'close', n = 252): ## 상기 close/ high / low / tr 컬럼 구조 그대로 따른다는 가정, df_return 은 daily
    ''' price = 'close', 'high', ' low, 'tr' '''
    ''' n = annualizing factor = must be chosen according to df_return interval'''

    if price not in ['close', 'high', 'low', 'tr']:

        raise ValueError("Price must be in close / high / low / tr")

    else:

        df_var = np.power(df_return[price], 2)

        daily = np.sqrt(n * df_var) # 1일 변동성
        ma5 = np.sqrt(n * df_var.rolling(5).sum() / 5).fillna(0) # 5일 변동성
        ma20 = np.sqrt(n * df_var.rolling(20).sum() / 20).fillna(0) # 20일 변동성
        wma_5 = np.sqrt(n * df_var.rolling(5).apply(lambda var_vector : np.dot(np.arange(1, 6), var_vector)) / np.arange(1, 6).sum()).fillna(0)
        wma_20 = np.sqrt(n * df_var.rolling(20).apply(lambda var_vector : np.dot(np.arange(1, 21), var_vector)) / np.arange(1, 21).sum()).fillna(0)
        ewma = df_var.ewm(alpha = 0.06, adjust = False).mean().pipe(lambda x : np.sqrt(x * n)).fillna(0)

        # garch derivation
        model = arch.arch_model(df_return[price] * 100, vol = 'garch', mean = 'zero')
        fit = model.fit(disp = "off")
        garch = np.sqrt(n) * fit.conditional_volatility / 100

        result = (daily + ma5 + ma20 + wma_5 + wma_20 + ewma + garch) / 7

    return result.dropna()


def frequency_table(ohlc_price, ub, lb, days, closeortr = 'close'):
    
    bins = np.linspace(ub, lb, 200)

    freq_list = dict()

    for i in range(days):
        df = ohlc_return(ohlc_price).total_return(i)[closeortr]
        df_freq = df.value_counts(bins = bins, normalize=  True)
        freq_list[i] = df_freq

    res = pd.DataFrame(freq_list)
    res.index = res.index.left
    
    return res