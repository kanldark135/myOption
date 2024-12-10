#%% 

import pandas as pd
import numpy as np
import compute as compute
import vol_forecast as myvf
import LEGACY.backtest as bt
import os

local_user = 'kanld'
path_xls = f"C:/Users/{local_user}/Desktop/"
path = os.getcwd()
path_pkl = path + "/data_pickle/"

# %% 
# 실현변동성 관련 지표 (volscore / 현재 분포에 기반한 임의 추정 / 시계열기반 GARCH 예측 / 머신러닝 기반 예측)

df_daily = pd.read_excel(path_xls + "종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'E:I').dropna()
df_daily = df_daily.iloc[:, 0:4].sort_index(ascending = True)
df_daily.index.name = 'date'
df_daily.columns = ['open','high','low','close']
df_daily.to_pickle(path_pkl + "/k200.pkl")

df_vkospi = pd.read_excel(path_xls + "종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'A:B').dropna()
df_vkospi = df_vkospi.sort_index(ascending = True)

a = myvf.vol_forecast(df_daily, 1)
b = myvf.vol_forecast(df_daily, 5)
c = myvf.vol_forecast(df_daily, 10)
d = myvf.vol_forecast(df_daily, 20)
e = myvf.vol_forecast(df_daily, 30)
f = myvf.vol_forecast(df_daily, 40)
g = myvf.vol_forecast(df_daily, 50)

table_volscore = pd.DataFrame([a.status()[0], b.status()[0], c.status()[0], d.status()[0], e.status()[0], f.status()[0], g.status()[0]], index = [1, 5, 10, 20, 30, 40, 50])
table_p = pd.DataFrame([a.status()[1], b.status()[1], c.status()[1], d.status()[1], e.status()[1], f.status()[1], g.status()[1]], index = [1, 5, 10, 20, 30, 40, 50])

#%% 내재변동성 관련

atm = 0.16

def iv_agg(monthlyweekly = 'monthly', callput = 'put', atm = None, dte = None):

    inst = bt.backtest(monthlyweekly, callput)
    moneyness_ub = 0
    moneyness_lb = 45

# 순전히 시계열 raw 데이터테이블

    datav = inst.iv_data(moneyness_ub, moneyness_lb, atm = atm, dte = dte)
    df_front = datav['front']
    df_back = datav['back']
    df_fskew = datav['fskew']
    df_bskew = datav['bskew']
    df_term = datav['term']

# 각 테이블에 대한 descriptive stats

    reportv = inst.iv_analysis(moneyness_ub, moneyness_lb, atm = atm, dte = dte)
    front = reportv['front']
    back = reportv['back']
    fskew = reportv['fskew']
    bskew = reportv['bskew']
    term = reportv['term']

    pd.concat([front, back, fskew, bskew, term], axis = 0).to_csv(f"./data_pickle/{monthlyweekly}_{callput}_result.csv")

    return datav, reportv

putm = iv_agg('monthly', 'put', atm = atm)
callm = iv_agg('monthly', 'call', atm = atm)
putw = iv_agg('weekly', 'put', dte = 6)
callw = iv_agg('weekly', 'call', dte = 6)


#%%
# weekly over monthly 별도계산

w = iv_agg('weekly', 'call', dte = 6)
m = iv_agg('monthly', 'call')
w_front = w[0]['front']
m_front = m[0]['front']
w_over_m = w_front.filter(regex = r'\d', axis = 1).divide(m_front.filter(regex = r'\d', axis = 1)).dropna(how = 'all')
w_over_m = w_over_m.merge(w_front[['dte', 'close', 'atm']], how = 'left', left_index = True, right_index = True)
res = w_over_m.describe()

dummy_list = []

for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    try:
        dummy = w_over_m.fillna(0).apply(lambda x : np.quantile(x, i)).to_frame(i).T
        dummy_list.append(dummy)
    except AttributeError:
        continue

w_over_m = pd.concat([res, *dummy_list])
w_over_m.to_csv("./data_pickle/woverm.csv")

# %%
