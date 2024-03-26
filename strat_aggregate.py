#%% 
import pandas as pd
import numpy as np
import scipy.optimize as sciop
#%%  월물
# 전략 추가/제거시 usecol 수정 필요

# df_monthly = pd.read_excel("./전략결과(240324).xlsx", sheet_name = 'total', usecols = "BF:DK", skiprows = [0])
df_monthly = pd.read_excel("./전략결과(240324).xlsx", sheet_name = 'total', usecols = "BF:DE", skiprows = [0])
n_size = 20 # 운용규모 커질수록 적절하게 조정 -> rounding 때문에...

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

weight = np.zeros(n_of_strats)

bound = [(0, n_size) for i in range(n_of_strats)] # no selling
const_1 = {'type' : 'eq', 'fun' : lambda w : np.sum(w) - n_size} # sum of all weight is 1

opt_result = sciop.minimize(obj_function, x0 = weight, args = (df_pnl), bounds = bound, constraints = [const_1])

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

#%% 주물

#1. 전략별로 loop 하는 preprocessing

df_weekly = pd.read_excel("./전략결과(240324).xlsx", sheet_name = 'total', usecols = "DM:FR", skiprows = [0])
n_size = 16 # 운용규모 커질수록 적절하게 조정

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

weight = np.zeros(n_of_strats)

bound = [(0, n_size) for i in range(n_of_strats)] # no selling
const_1 = {'type' : 'eq', 'fun' : lambda w : np.sum(w) - n_size} # sum of all weight is 15

opt_result = sciop.minimize(obj_function, x0 = weight, args = (df_pnl), bounds = bound, constraints = [const_1])

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
