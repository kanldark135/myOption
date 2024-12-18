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

df_monthly = pd.read_excel("./전략결과(240530~).xlsx", sheet_name = 'total', usecols = "BO:DP", skiprows = [0])
principal = 150000000 / 250000
n_size = 100
optimal_multiplier = 0.2

#1. 전략별로 loop 하는 preprocessing

n_of_strats = int(df_monthly.shape[1] / 2)

df_pnl = pd.DataFrame()

for i in range(n_of_strats):
    df_dummy = df_monthly.iloc[:, 2 * i: 2 * i + 2]
    df_dummy = df_dummy.iloc[1:]
    
    strat_name = df_dummy.columns[0]
    df_dummy = df_dummy.dropna(axis=0, subset=[strat_name], how='all')
    
    df_dummy = df_dummy.set_index(strat_name)
    df_dummy.columns = [strat_name]
    df_dummy.index.name = 'date'
    
    dummy_date = pd.date_range('2010-01-01', '2023-07-31')
    df_dummy = df_dummy.reindex(dummy_date)
    df_dummy.apply(pd.to_numeric, errors='coerce')
    
    df_dummy = df_dummy.fillna(method='ffill')
    
    df_pnl = pd.concat([df_pnl, df_dummy], axis=1)

# Drawdown calculation
df_drawdown = df_pnl.apply(lambda x: x - x.cummax(), axis=0)

# Objective function definition
def obj_function_1(weight, df_pnl):
    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis=1).sum(axis=1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    
    max_pnl = weighted_pnl.max()
    max_drawdown = min(weighted_drawdown.min(), -0.0001)  # prevent division by zero error
    
    res = max_pnl / max_drawdown
    
    return res

def obj_function_2(weight, df_pnl, principal, max_what = 'sharpe'):
    
    ''' 전략의 일일 손익데이터가 수익률이 아니라 수익금인 만큼
    계산의 편의를 위해 전략에 대한 재투자는 없이 버는돈 족족 인출한다는 가정임'''

    principal = principal / 250000 # 금액 -> 지수로 환산

    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis=1).sum(axis=1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    weighted_daily_return = (weighted_pnl - weighted_pnl.shift(1)) / principal

    daily_avg_return = weighted_daily_return.mean()
    daily_std = weighted_daily_return.std()
    daily_std_negative = weighted_daily_return.loc[weighted_daily_return > 0].std()

    total_years = (weighted_daily_return.index[-1] - weighted_daily_return.index[0]).days / 365
    total_return = weighted_pnl.iloc[-1] / principal - 1
    annual_return = total_return / total_years
    max_pnl = weighted_pnl.max()
    max_loss = min(weighted_drawdown.min(), -0.0001)  # prevent division by zero error
    max_drawdown = max_loss / principal

    res_dict = dict(
    max_pnl = max_pnl / max_loss,
    sharpe = - np.sqrt(252) * daily_avg_return / daily_std, # turn negative for minimization
    sortino = - np.sqrt(252) * daily_avg_return / daily_std_negative, # turn negative for minimization
    calmar = annual_return / max_drawdown
    )

    res =  res_dict[max_what]
    return res

# Initial weight
weight = np.ones(n_of_strats) / n_of_strats

# Bounds for the weights
# bound = [(0, 0.1) for _ in range(n_of_strats)]
bound = [(0, 0.08) for _ in range(n_of_strats)]

# Constraints
const = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # sum of all weight should be >= n_lower

# Minimize the objective function
# opt_result = sciop.minimize(obj_function_1, x0=weight, args=(df_pnl), bounds=bound, constraints=[const])
opt_result = sciop.minimize(obj_function_2, x0=weight, args=(df_pnl, principal, 'calmar'), bounds=bound, constraints=[const], method = 'SLSQP')

print(obj_function_2(opt_result.x, df_pnl, principal, 'sharpe'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'sortino'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'calmar'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'max_pnl'))

pnl = df_pnl.multiply(opt_result.x).sum(axis = 1)
dd = pnl - pnl.cummax()
ratio = pnl.max() / dd.min()
res = pd.DataFrame(data = opt_result.x, index = df_pnl.columns, columns = ['weight'])
print(res)
print(pnl ,dd, ratio)

pnl_int = df_pnl.multiply(np.round(n_size * opt_result.x)).sum(axis = 1)
dd_int = pnl_int - pnl_int.cummax()
ratio_int = pnl_int.max() / dd_int.min()

res_int = pd.DataFrame(data = np.round(n_size * opt_result.x), index = df_pnl.columns, columns = ['weight'])

res_int.to_csv("./res.csv")
pnl_int.to_csv("./ret.csv")


#%% 주물

#1. 전략별로 loop 하는 preprocessing

df_weekly = pd.read_excel("./전략결과(240530~).xlsx", sheet_name = 'total', usecols = "DR:FC", skiprows = [0])

n_size = 100
optimal_multiplier = 0.91

#1. 전략별로 loop 하는 preprocessing

n_of_strats = int(df_weekly.shape[1] / 2)

df_pnl = pd.DataFrame()

for i in range(n_of_strats):
    df_dummy = df_weekly.iloc[:, 2 * i: 2 * i + 2]
    df_dummy = df_dummy.iloc[1:]
    
    strat_name = df_dummy.columns[0]
    df_dummy = df_dummy.dropna(axis=0, subset=[strat_name], how='all')
    
    df_dummy = df_dummy.set_index(strat_name)
    df_dummy.columns = [strat_name]
    df_dummy.index.name = 'date'
    
    dummy_date = pd.date_range('2019-01-01', '2023-07-31')
    df_dummy = df_dummy.reindex(dummy_date)
    df_dummy.apply(pd.to_numeric, errors='coerce')
    
    df_dummy = df_dummy.fillna(method='ffill')
    
    df_pnl = pd.concat([df_pnl, df_dummy], axis=1)

# Drawdown calculation
df_drawdown = df_pnl.apply(lambda x: x - x.cummax(), axis=0)

# Objective function definition
def obj_function_1(weight, df_pnl):
    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis=1).sum(axis=1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    
    max_pnl = weighted_pnl.max()
    max_drawdown = min(weighted_drawdown.min(), -0.0001)  # prevent division by zero error
    
    res = max_pnl / max_drawdown
    
    return res

def obj_function_2(weight, df_pnl, principal, max_what = 'sharpe'):
    
    ''' 전략의 일일 손익데이터가 수익률이 아니라 수익금인 만큼
    계산의 편의를 위해 전략에 대한 재투자는 없이 버는돈 족족 인출한다는 가정임'''

    principal = principal / 250000 # 금액 -> 지수로 환산

    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis=1).sum(axis=1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    weighted_daily_return = (weighted_pnl - weighted_pnl.shift(1)) / principal

    daily_avg_return = weighted_daily_return.mean()
    daily_std = weighted_daily_return.std()
    daily_std_negative = weighted_daily_return.loc[weighted_daily_return > 0].std()

    total_years = (weighted_daily_return.index[-1] - weighted_daily_return.index[0]).days / 365
    total_return = weighted_pnl.iloc[-1] / principal - 1
    annual_return = total_return / total_years
    max_pnl = weighted_pnl.max()
    max_loss = min(weighted_drawdown.min(), -0.0001)  # prevent division by zero error
    max_drawdown = max_loss / principal

    res_dict = dict(
    max_pnl = max_pnl / max_loss,
    sharpe = - np.sqrt(252) * daily_avg_return / daily_std, # turn negative for minimization
    sortino = - np.sqrt(252) * daily_avg_return / daily_std_negative, # turn negative for minimization
    calmar = annual_return / max_drawdown
    )

    res =  res_dict[max_what]
    return res

# Initial weight
weight = np.ones(n_of_strats) / n_of_strats

# Bounds for the weights
# bound = [(0, 0.1) for _ in range(n_of_strats)]
bound = [(0, 0.08) for _ in range(n_of_strats)]

# Constraints
const = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # sum of all weight should be >= n_lower

# Minimize the objective function
# opt_result = sciop.minimize(obj_function_1, x0=weight, args=(df_pnl), bounds=bound, constraints=[const])
opt_result = sciop.minimize(obj_function_2, x0=weight, args=(df_pnl, principal, 'calmar'), bounds=bound, constraints=[const], method = 'SLSQP')

print(obj_function_2(opt_result.x, df_pnl, principal, 'sharpe'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'sortino'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'calmar'))
print(obj_function_2(opt_result.x, df_pnl, principal, 'max_pnl'))

pnl = df_pnl.multiply(opt_result.x).sum(axis = 1)
dd = pnl - pnl.cummax()
ratio = pnl.max() / dd.min()
res = pd.DataFrame(data = opt_result.x, index = df_pnl.columns, columns = ['weight'])
print(res)
print(pnl ,dd, ratio)

pnl_int = df_pnl.multiply(np.round(n_size * opt_result.x)).sum(axis = 1)
dd_int = pnl_int - pnl_int.cummax()
ratio_int = pnl_int.max() / dd_int.min()

res_int = pd.DataFrame(data = np.round(n_size * opt_result.x), index = df_pnl.columns, columns = ['weight'])

res_int.to_csv("./res.csv")
pnl_int.to_csv("./ret.csv")

#%% loop over 
# multiple initial values
# with 0 ~ 1 weight bound for each strategy (no leverage)
# constraint 1 =  the sum of the weights of all strategy = 1

n_size = 100

# df_ret = pd.read_excel("./전략결과(240530).xlsx", sheet_name = 'total', usecols = "BO:DL", skiprows = [0])
df_ret = pd.read_excel("./전략결과(240530).xlsx", sheet_name = 'total', usecols = "DN:FE", skiprows = [0])

n_of_strats = int(df_ret.shape[1] / 2)

df_pnl = pd.DataFrame()

for i in range(n_of_strats):
    df_dummy = df_ret.iloc[:, 2 * i: 2 * i + 2]
    df_dummy = df_dummy.iloc[1:]
    
    strat_name = df_dummy.columns[0]
    df_dummy = df_dummy.dropna(axis=0, subset=[strat_name], how='all')
    
    df_dummy = df_dummy.set_index(strat_name)
    df_dummy.columns = [strat_name]
    df_dummy.index.name = 'date'
    
    dummy_date = pd.date_range('2010-01-01', '2023-07-31')
    df_dummy = df_dummy.reindex(dummy_date)
    df_dummy.apply(pd.to_numeric, errors='coerce')
    
    df_dummy = df_dummy.fillna(method='ffill')
    
    df_pnl = pd.concat([df_pnl, df_dummy], axis=1)

# Drawdown calculation
df_drawdown = df_pnl.apply(lambda x: x - x.cummax(), axis=0)

# Objective function definition
def obj_function(weight, df_pnl):
    if len(weight) != df_pnl.shape[1]:
        raise IndexError("weight size not matching size of the pnl")
    weighted_pnl = df_pnl.multiply(weight, axis=1).sum(axis=1)
    weighted_drawdown = weighted_pnl - weighted_pnl.cummax()
    
    max_pnl = weighted_pnl.max()
    max_drawdown = min(weighted_drawdown.min(), -0.0001)  # prevent division by zero error
    
    res = max_pnl / max_drawdown
    
    return res

# Initial weight
weight = np.ones(n_of_strats)
# Bounds for the weights
bound = [(0, 1) for _ in range(n_of_strats)]
const = {'type' : 'eq', 'fun' : lambda w : np.sum(w) - 1} 

w_list = np.linspace(0, 1, 101) # initial value 에 따른 차이

df_res = pd.DataFrame(columns = ['ratio', 'ratio_int'])
count = 0

for i in w_list:

    opt_result = sciop.minimize(obj_function, x0 = weight * i, args = (df_pnl), bounds = bound, constraints = [const])

    try:
        pnl = df_pnl.multiply(opt_result.x).sum(axis = 1)
        dd = pnl - pnl.cummax()
        ratio = pnl.max() / dd.min()
        
    except ZeroDivisionError:
        ratio = pnl.max() / 0.0001
    
    try:
        pnl_int = df_pnl.multiply(np.round(n_size * opt_result.x)).sum(axis = 1)
        dd_int = pnl_int - pnl_int.cummax()
        ratio_int = pnl_int.max() / dd_int.min()
        
    except ZeroDivisionError:
        ratio_int = pnl_int.max() / 0.0001

    count += i
    print(count)

    df_res.loc[i] = [ratio, ratio_int]
    
    