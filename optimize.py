import pandas as pd
import numpy as np
import get_entry_exit
import datetime
import time
import joblib
import matplotlib.pyplot as plt
import backtest as bt
import pathlib
import bayes_opt 

db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_option.db")

df_k200 = bt.get_timeseries(db_path, "k200")['k200']
entry_dates = df_k200.weekday(0)
table = 'monthly'

call1 = {'entry_dates' : entry_dates, 'table' : 'weekly_mon', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : -1}
put1 = {'entry_dates' : entry_dates, 'table' : 'weekly_mon', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : -1}

var = dict( 
stop_dates = [],
dte_stop = 1,
profit = 0.5,
loss = -0.5,
is_complex_pnl = False,
is_intraday_stop = False,
start_date = '20100101',
end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
show_chart = False
)

res = bt.backtest(call1, put1).equal_inout(**var)['pnl']

def objective(profit, loss):
    var['profit'] = profit
    var['loss'] = loss

    result = bt.backtest(call1).equal_inout(**var)['pnl']
    ratio = np.abs(result['cum_pnl'].max() / result['dd'].min())
    return ratio

pbounds = {
    'profit': (0.0, 1),  # profit의 범위
    'loss': (-999, 0)   # loss의 범위
}
# Bayesian Optimization 실행
optimizer = bayes_opt.BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42
)

# 최적화 시작
optimizer.maximize(
    init_points=5,  # 초기 랜덤 탐색 횟수
    n_iter=25       # Bayesian Optimization 반복 횟수
)

# 최적의 파라미터와 결과
print("Optimal Parameters:", optimizer.max['params'])
print("Best PnL:", optimizer.max['target'])
