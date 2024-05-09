#%% 
import pandas as pd
import numpy as np
import scipy.optimize as sciop

def custom_res(df_pnl, custom_weight):
    pnl = df_pnl.multiply(custom_weight).sum(axis = 1)
    dd = (pnl - pnl.cummax()).min()
    ratio = pnl.max() / dd

    return pnl, dd, ratio


#%%  월물
# 전략 추가/제거시 usecol 수정 필요

# df_monthly = pd.read_excel("./전략결과(240324).xlsx", sheet_name = 'total', usecols = "BF:DK", skiprows = [0])
df_monthly = pd.read_excel("./전략결과_2(240509).xlsx", sheet_name = 'total', usecols = "BB:DK", skiprows = [0])
n_lower = 0 # 운용규모 커질수록 적절하게 조정 -> rounding 때문에...
n_higher = 60

#1. 전략별로 loop 하는 preprocessing

n_of_strats = int(df_monthly.shape[1] / 2)

df_pnl = pd.DataFrame()

for i in range(n_of_strats):

    df_dummy = df_monthly.iloc[:, 2*i : 2*i + 2]
    df_dummy = df_dummy.iloc[1:]

    strat_name = df_dummy.columns[0]
    df_dummy = df_dummy.dropna(axis = 0, subset = [strat_name], how = 'all')

    df_dummy = df_dummy.set_index(strat_name)
    df_dummy.columns = [strat_name]
    df_dummy.index.name = 'date'

    dummy_date = pd.date_range('2010-01-01', '2023-07-31')
    df_dummy = df_dummy.reindex(dummy_date)
    df_dummy.apply(pd.to_numeric, errors = 'coerce')

    df_dummy = df_dummy.fillna(method = 'ffill')

    df_pnl = pd.concat([df_pnl, df_dummy], axis = 1)

df_drawdown = df_pnl.apply(lambda x : x - x.cummax(), axis = 0)

def obj_function(weight, df_pnl):
    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis = 1).sum(axis = 1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()

    max_pnl = weighted_pnl.max()
    max_drawdown = min(weighted_drawdown.min(), -0.0001) # prevent divisionbyzero error

    res = max_pnl / max_drawdown

    return res

weight = np.ones(n_of_strats)

bound = [(0, n_higher) for i in range(n_of_strats)] # no selling
const_1 = {'type' : 'ineq', 'fun' : lambda w : np.sum(w) - n_lower} # sum of all weight is 1
const_2 = {'type' : 'ineq', 'fun' : lambda w : n_higher - np.sum(w)} # sum of all weight is 1

opt_result = sciop.minimize(obj_function, x0 = weight, args = (df_pnl), bounds = bound, constraints = [const_1, const_2])

pnl = df_pnl.multiply(opt_result.x).sum(axis = 1)
dd = pnl - pnl.cummax()
ratio = pnl.max() / dd.min()

res = pd.DataFrame(data = opt_result.x, index = df_pnl.columns, columns = ['weight'])
print(res)
print(pnl ,dd, ratio)

pnl_int = df_pnl.multiply(np.round(opt_result.x)).sum(axis = 1)
dd_int = pnl_int - pnl_int.cummax()
ratio_int = pnl_int.max() / dd_int.min()

res_int = pd.DataFrame(data = np.round(opt_result.x), index = df_pnl.columns, columns = ['weight'])

res_int.to_csv("./res.csv")
pnl_int.to_csv("./ret.csv")

#%% 주물

#1. 전략별로 loop 하는 preprocessing

df_weekly = pd.read_excel("./전략결과(240324).xlsx", sheet_name = 'total', usecols = "DG:FL", skiprows = [0])

n_lower = 0
n_higher = 50 # 운용규모 커질수록 적절하게 조정

n_of_strats = int(df_weekly.shape[1] / 2)
df_pnl = pd.DataFrame()

for i in range(n_of_strats):

    df_dummy = df_weekly.iloc[:, 2*i : 2*i + 2]
    df_dummy = df_dummy.iloc[1:]

    strat_name = df_dummy.columns[0]
    df_dummy = df_dummy.dropna(axis = 0, subset = [strat_name], how = 'all')

    df_dummy = df_dummy.set_index(strat_name)
    df_dummy.columns = [strat_name]
    df_dummy.index.name = 'date'

    dummy_date = pd.date_range('2019-01-01', '2023-07-31')
    df_dummy = df_dummy.reindex(dummy_date)
    df_dummy.apply(pd.to_numeric, errors = 'coerce')

    df_dummy = df_dummy.fillna(method = 'ffill')

    df_pnl = pd.concat([df_pnl, df_dummy], axis = 1)

df_drawdown = df_pnl.apply(lambda x : x - x.cummax(), axis = 0)

def obj_function(weight, df_pnl):
    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis = 1).sum(axis = 1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    
    max_pnl = weighted_pnl.max()
    max_drawdown = min(weighted_drawdown.min(), -0.0001) # prevent divisionbyzero error

    res = max_pnl / max_drawdown

    return res

weight = np.ones(n_of_strats)

bound = [(0, n_higher) for i in range(n_of_strats)] # no selling
const_1 = {'type' : 'ineq', 'fun' : lambda w : np.sum(w) - n_lower} # sum of all weight is 15
const_2 = {'type' : 'ineq', 'fun' : lambda w : n_higher - np.sum(w)} # sum of all weight is 15

opt_result = sciop.minimize(obj_function, x0 = weight, args = (df_pnl), bounds = bound, constraints = [const_1, const_2])

try:
    pnl = df_pnl.multiply(opt_result.x).sum(axis = 1)
    dd = pnl - pnl.cummax()
    ratio = pnl.max() / dd.min()
except ZeroDivisionError:
    ratio = pnl.max() / 0.0001

res = pd.DataFrame(data = opt_result.x, index = df_pnl.columns, columns = ['weight'])
print(res)
print(pnl ,dd, ratio)

try:
    pnl_int = df_pnl.multiply(np.round(opt_result.x)).sum(axis = 1)
    dd_int = pnl_int - pnl_int.cummax()
    ratio_int = pnl_int.max() / dd_int.min()
except ZeroDivisionError:
    ratio_int = pnl_int.max() / 0.0001

res_int = pd.DataFrame(data = np.round(opt_result.x), index = df_pnl.columns, columns = ['weight'])

res_int.to_csv("./res.csv")
pnl_int.to_csv("./ret.csv")

