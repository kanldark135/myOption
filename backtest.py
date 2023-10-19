#%% 
import pandas as pd
import numpy as np
import compute as compute

# %% 데이터 추출 클래스

class backtest:

    ''' period = monthly / weekly, callput = 'call / put'''

    def __init__(self, monthlyorweekly = 'monthly', callput = 'put'):

        route = f"./data_pickle/{callput}_{monthlyorweekly}.pkl"

        self.df = pd.read_pickle(route) # 전체
        self.df_front = self.df[self.df['cycle'] == 0] # 근월물
        self.df_back = self.df[self.df['cycle'] == 1] # 차월물
        self.df_backback = self.df[self.df['cycle'] == 2] # 차차월물
        
        if monthlyorweekly == 'monthly':
            self.remove_dte = list(range(0, 9))
        else:
            self.remove_dte = list(range(9, 1000))

    def to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]
        
    def atm_and_dte(self, df, atm = None, dte = None):

        def similar_atm(df, atm, buffer = 0.01):
            cond_1 = df['atm'] >= atm - buffer
            cond_2 = df['atm'] <= atm + buffer
            df_similar_atm = df[cond_1 & cond_2]
            return df_similar_atm

        def similar_dte(df, dte, buffer = 1):
            cond_1 = df['dte'] >= dte - buffer
            cond_2 = df['dte'] <= dte + buffer
            df_similar_dte = df[cond_1 & cond_2]
            return df_similar_dte
        
        if atm != None:
            if dte != None:
                res = df.pipe(similar_atm, atm).pipe(similar_dte, dte)
            else:
                res = similar_atm(df, atm)
        else:
            if dte != None:
                res = similar_dte(df, dte)
            else:
                res = df        
        return res
        
    def get_data(self, component = 'iv', frontback = 'front', moneyness_lb = 0, moneyness_ub = 30, atm = None, dte = None):
        '''components : iv, price, delta, gamma, theta, vega'''

        if frontback == 'front':
            df = self.df_front
        elif frontback == 'back':
            df = self.df_back
        else:
            df = self.df
        
        cond = {'cond1' : df['moneyness'] >= moneyness_lb, 
                'cond2' : df['moneyness'] <= moneyness_ub,
                'cond3' : ~df['dte'].isin(self.remove_dte)
                }
        df_cond = df[
        (cond['cond1'] & cond['cond2'] & cond['cond3'])
        ]

        bool_unique = ~df_cond.index.duplicated()

        res = df_cond.pivot_table(values = component, index = df_cond.index, columns = 'moneyness')
        res = res.merge(df_cond[['dte', 'close']].loc[bool_unique], how = 'left', left_on = res.index , right_index = True)
        res['atm'] = res[0]

        res = self.atm_and_dte(res, atm = atm, dte = dte)

        return res
    
    def iv_data(self, moneyness_lb = 0, moneyness_ub = 30, atm = None, dte = None):
    
    # 1) raw iv data

        df_front = self.get_data(component = 'iv', frontback = 'front', moneyness_lb = moneyness_lb, moneyness_ub = moneyness_ub, atm = atm, dte = dte)
        df_back = self.get_data(component = 'iv', frontback = 'back', moneyness_lb = moneyness_lb, moneyness_ub = moneyness_ub, atm = atm, dte = dte)

        cut_front = df_front.loc[:, moneyness_lb : moneyness_ub]
        cut_back = df_back.loc[:, moneyness_lb : moneyness_ub]

# 핵심 계산 ---------------------------------------------------------------------

    # 2) skewness over atm

        front_skew = cut_front \
            .apply(lambda x : np.divide(x, x[0]), axis = 1)\
            .dropna(how = 'all')\
            .merge(df_front[['atm', 'dte', 'close']], how = 'left', left_index = True, right_index = True)
        
        back_skew = cut_back \
            .apply(lambda x : np.divide(x, x[0]), axis = 1)\
            .dropna(how = 'all')\
            .merge(df_front[['atm', 'dte', 'close']], how = 'left', left_index = True, right_index = True)

    # 3) term spread, dte 는 front 기준
    # back 에 있는 0 때문에 발생하는 inf 문제 (저 0은 데이터 정리할때 nan 값 안받는 함수때문에 어쩔수없이 nan -> 0 으로 바꾼거임)
        
        front_over_back = cut_front.divide(cut_back)
        front_over_back = front_over_back \
            .replace(np.inf, np.nan) \
            .dropna(how = 'all') \
            .merge(df_front[['atm', 'dte', 'close']], how = 'left', left_index = True, right_index = True)

# ---------------------------------------------------------------------

        res = {
            'front' : df_front,
            'back' : df_back, 
            'fskew' : front_skew,
            'bskew' : back_skew,
            'term' : front_over_back
            }
        
        return res
    
    def iv_analysis(self, moneyness_lb = 0, moneyness_ub = 30, atm = None, dte = None):

        '''유사 atm IV 수준만 골라서 비교하려면 atm 에 값 기록'''
        ''''''
        data = self.iv_data(moneyness_lb, moneyness_ub, atm, dte)

        res = {}

        # 1) descriptive stats

        for key in data.keys():

            df = data.get(key)

            # n/a left as n/a

            stats = df.describe()
            df_quantile = df.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], axis = 0, interpolation = 'linear')
            
            stats = pd.concat([stats, df_quantile])
            res[key] = stats

        return res

## 이제 할 거

# 전반적으로 좀 더 보기 쉽게 만들기
# 가격테이블 가져오면서 backtesting 툴 만들기

# %%

# import backtest as bt

# a = bt.backtest()

# data = a.iv_data()

# front = data['front']
# back = data['back']
# fskew = data['fskew']
# bskew = data['bskew']
# term = data['term']

# report = a.iv_analysis(quantile = 0.1)

# %%
