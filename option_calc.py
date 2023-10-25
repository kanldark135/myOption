# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:19:21 2022

@author: Kwan
"""
import pandas as pd
import numpy as np
import scipy.stats as scistat
import scipy.optimize as sciop

    
def call_p(s, k, v, t, r, q = 0):
    
    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        N_d1 = scistat.norm.cdf(d1)
        d2 = d1 - v * np.sqrt(t)
        N_d2 = scistat.norm.cdf(d2)
        
        price = s *np.exp(-q * t) * N_d1 - k * np.exp(-r * t) * N_d2
    
    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            price = np.nan
        else:
            raise e
        
    return price

def put_p(s, k, v, t, r, q = 0):

    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        N_d1 = scistat.norm.cdf(d1)
        d2 = d1 - v * np.sqrt(t)
        N_d2 = scistat.norm.cdf(d2)
        
        price = k * np.exp(-r * t) * (1 - N_d2) - s * np.exp(-q * t) * (1 - N_d1)

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            price = np.nan
        else:
            raise e
        
    return price

def call_delta(s, k, v, t, r, q = 0):
    
    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        N_d1 = scistat.norm.cdf(d1)

        delta = N_d1

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            delta = np.nan
        else:
            raise e
        
    return delta
    
def put_delta(s, k, v, t, r, q = 0):

    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        N_d1 = scistat.norm.cdf(d1)

        delta = - (1 - N_d1)

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            delta = np.nan
        else:
            raise e
        
    return delta


def gamma(s, k, v, t, r, q = 0):
    
    try: 

        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        n_d1 = scistat.norm.pdf(d1)
        
        gamma = n_d1 * np.exp(-q * t) / (s * v * np.sqrt(t))

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            gamma = np.nan
        else:
            raise e
        
    return gamma

def vega(s, k, v, t, r, q = 0):

    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        n_d1 = scistat.norm.pdf(d1)
        
        vega = (s * np.exp(-q * t) * np.sqrt(t) * n_d1)/100

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            vega = np.nan
        else:
            raise e
    
    return vega
    
def call_theta(s, k, v, t, r, q = 0):

    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        
        theta = - (s * np.exp(-q * t) * v * scistat.norm.pdf(d1) / (2 * np.sqrt(t))) - (r * k * np.exp(-r * t) * scistat.norm.cdf(d2)) + (q * s * np.exp(-q * t) * scistat.norm.cdf(d1))
        call_theta = theta/365
    
    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            call_theta = np.nan
        else:
            raise e

    return call_theta

def put_theta(s, k, v, t, r, q = 0):

    try:
    
        d1 = (np.log(s / k) + (r - q + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        
        theta = - (s * np.exp(-q * t) * v * scistat.norm.pdf(d1) / (2 * np.sqrt(t))) + (r * k * np.exp(-r * t) * scistat.norm.cdf(-d2)) - (q * s * np.exp(-q * t) * scistat.norm.cdf(-d1))
        put_theta = theta/365

    except TypeError as e:
        if any(np.isnan([s, k, v, t, r, q])):
            put_theta = np.nan
        else:
            raise e
        
    return put_theta
    
    
## 가격 = spot 이 주어졌을때, 개별옵션의 IV 역산하여 도출, minimizing Least Square 로 접근하였음

def derive_iv(s, k, v, t, r, spot, q = 0, callput = 'call'):
    
    iv = 0
    
    if callput == 'call':
        
        def diff_func_LS(v, s, k, t, r, spot, q):
            result = np.sum(np.power(call_p(s, k, v, t, r, q) - spot, 2))
            return result
        
        optimize = sciop.minimize(diff_func_LS, x0 = 0.1, args = (s, k, t, r, spot, q))
        iv = optimize.x
        
    else: 
        
        def diff_func_LS(v, s, k, t, r, spot, q):
            result = np.sum(np.power(put_p(s, k, v, t, r, q) - spot, 2))
            return result
        
        optimize = sciop.minimize(diff_func_LS, x0 = 0.1, args = (s, k, t, r, spot, q))
        iv = optimize.x
        
    return iv
    
## 헤지스킴 도출 목적으로 주가 각 구간별로 내 현재 포지션의 가격 및 델타 예상치에 대한 df 만들어내는 특수 함수
## 나중에 포지션 projection 툴이랑 엮어서 수정할 예정
    
def generate_hedge_df(strike, hedge_interval, s, v, t, r, number, callput = 'call'):
    
    s_array = [s + i * hedge_interval for i in range(-20, 20)]
    v_array = np.repeat(v, len(s_array))
    
    # 주가별로 IV 동일하다는 가정은 사실 비현실적. 주가변화에 따른 개별옵션의 IV도 변화 => IV Curve 변형 없다는 가정 하 IV Curve 가져온 뒤 realized IV 식으로 array 구성
    
    df =  pd.DataFrame(dict(s = s_array, k = strike, v = v_array, t = t, r = r), index = s_array)
    
    if callput == 'call':
        df = df.assign(p = call_p(df.s, df.k, df.v, df.t, df.r), position = call_p(df.s, df.k, df.v, df.t, df.r) * number, delta = call_delta(df.s, df.k, df.v, df.t, df.r), position_delta = call_delta(df.s, df.k, df.v, df.t, df.r) * number)        
    else:
        df = df.assign(p = put_p(df.s, df.k, df.v, df.t, df.r), position = put_p(df.s, df.k, df.v, df.t, df.r) * number, delta = put_delta(df.s, df.k, df.v, df.t, df.r), position_delta = put_delta(df.s, df.k, df.v, df.t, df.r) * number)
 
    return df

def get_option_data(component = 'iv', monthlyorweekly = 'monthly', cycle = 'front', callput = 'put', 
                    moneyness_lb = 0, moneyness_ub = 30, atm_range:list = None, dte_range:list = None):
    
    ''' 데이터 pkl 경로 관련 변수 추가'''
    '''components : iv, price, delta, gamma, theta, vega'''

    # 데이터 pkl 로컬에 있어야 함
    
    route = f"./data_pickle/{monthlyorweekly}.pkl"
    # dict_cycle = {'front' : 0,
    #          'back' : 1,
    #          'backback' : 2}
    
    df = pd.read_pickle(route) # 전체
    # df = df[df['cycle'] == dict_cycle.get(cycle)] # 1. 근/차월물 구분

    cond = {'cond1' : df['moneyness'] >= moneyness_lb, 'cond2' : df['moneyness'] <= moneyness_ub}
    df = df[(cond['cond1'] & cond['cond2'])] # 2. 원하는 moneyness 선택
    
    bool_unique = ~df.index.duplicated()

    res = df.pivot_table(values = component, index = df.index, columns = 'moneyness')
    res = res.merge(df[['dte','expiry', 'close']].loc[bool_unique], how = 'left', left_on = res.index , right_index = True)
    res['atm'] = res[0]

    def atm_and_dte(df, atm = None, dte = None):

        def atm_range(df, atm:list):
            cond_1 = df['atm'] >= np.min(atm)
            cond_2 = df['atm'] < np.max(atm)
            df_similar_atm = df[cond_1 & cond_2]
            return df_similar_atm

        def dte_range(df, dte:list):
            cond_1 = df['dte'] >= np.min(dte)
            cond_2 = df['dte'] < np.max(dte)
            df_similar_dte = df[cond_1 & cond_2]
            return df_similar_dte
        
        if atm != None:
            if dte != None:
                res = df.pipe(atm_range, atm).pipe(dte_range, dte)
            else:
                res = atm_range(df, atm)
        else:
            if dte != None:
                res = dte_range(df, dte)
            else:
                res = df        
        return res
    
    res = atm_and_dte(res, atm = atm_range, dte = dte_range)     # 3. 원하는 ATM의 값 및 DTE
    return res

def get_skewness(df):
    copy = df.filter(regex = r"\d")
    copy = copy.apply(lambda x : x / df['atm'])
    res = copy.combine_first(df)
    return res

def get_calendar(df_front, df_back):
    copy_1 = df_front.filter(regex = r'\d')
    copy_2 = df_back.filter(regex = r'\d')
    res = copy_2 / copy_1
    res = res.dropna(how = 'all')
    res = res.merge(df_front[['dte', 'expiry', 'close','atm']], how = 'left', right_index = True, left_index = True)  # 잔존기간은 근월물 기준으로 merge
    return res

def get_closest_strike(close_price):

    divided = divmod(close_price, 2.5)
    if divided[1] > 1.25:
        res = (divided[0] + 1) * 2.5
    else:
        res = divided[0] * 2.5 

    return res