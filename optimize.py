import pandas as pd
import numpy as np
import get_entry_exit
import datetime
import time
import joblib
import matplotlib.pyplot as plt
import backtest as bt
import pathlib
import itertools

def generate_iterables(*args):
    container = []
    for item in args:
        if not isinstance(item, (list, tuple, dict, np.ndarray)):
            container.append([item])
        else:
            container.append(item)
            
    res = itertools.product(*container)
    return res

class metrics:

    def __new__(cls, df_result):
        instance = super().__new__(cls)
        instance.df = df_result
        res = instance.calculate_metrics()
        return res
        
    def calculate_metrics(self):

        basic_metrics = self.basic(self.df)
        sortino_metrics = self.sortino(self.df)
        calmar_metrics = self.calmar(self.df)
        avg_profit_loss_metrics = self.avg_profit_loss(self.df)
        win_count_metrics = self.win_count(self.df)
        
        # 모든 메트릭 결과를 병합
        all_metrics = {**basic_metrics, **sortino_metrics, **calmar_metrics, 
                       **avg_profit_loss_metrics, **win_count_metrics}
        
        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame([all_metrics])
        
        return result_df

    def basic(self, df):

        pnl = df['pnl']

        res_dict = dict(
        cum_ret = pnl['cum_pnl'].iloc[-1],
        max_ret = pnl['cum_pnl'].max(),
        mdd = pnl['dd'].min()
        )

        return res_dict
        
    def sortino(self, df):
        pnl = df['pnl']
        years = (pnl['cum_pnl'].index[-1] - pnl['cum_pnl'].index[1]) / datetime.timedelta(days = 365)
        annual_gain = pnl['cum_pnl'].iloc[-1] / years
        downside_std = np.sqrt(252) * pnl['daily_pnl'].loc[pnl['daily_pnl'] < 0].std()

        ratio = annual_gain / downside_std
        return dict(sortino = ratio)

    def calmar(self, df):
        '''
        최소 1이상은 나와줘야
        '''

        pnl = df['pnl']
        years = (pnl['cum_pnl'].index[-1] - pnl['cum_pnl'].index[1]) / datetime.timedelta(days = 365)
        annual_gain = pnl['cum_pnl'].iloc[-1] / years
        mdd = pnl['dd'].min()

        ratio = abs(annual_gain / mdd)
        return dict(calmar = ratio)

    def avg_profit_loss(self, df):
        '''
        1 이상이면 좋은거
        '''
        res = df['res']
        avg_profit = res['cum_pnl'].loc[res['cum_pnl'] > 0].mean()
        avg_loss = res['cum_pnl'].loc[res['cum_pnl'] < 0].mean()

        ratio = abs(avg_profit / avg_loss)
        return dict(avg_pnl_ratio = ratio)
    
    def win_count(self, df):
        res = df['res'].value_counts('whystop')
        
        res_dict = dict(
        dte = res.get('dte', 0),
        win = res.get('win', 0),
        loss = res.get('loss', 0),
        stop = res.get('stop', 0)
        )

        return res_dict
    
''' query_var 랑 trade_var 랑 다르게 만들어야 함 query_var는 여러 leg 동시에 받으므로 일률적으로 select_value 적용 불가'''


db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/weekonly.db")

# entry 조건
df_k200 = bt.get_timeseries(db_path, "k200")['k200']
entry_dates = df_k200.weekday(4)
# 쿼리조건
# 다른 조건은 다 밑에서 전술적으로 조정 : 가령 monthly call 매수 / weekly call 매도인 경우 일률적으로 iterable 만들 수 없음
table = 'weekly_thu'
type = 'moneyness'
select_values = [5, 7.5, 10]

query_vars = generate_iterables(entry_dates, table, type, select_values) 

call1 = {'entry_dates' : entry_dates,
            'table' : table,
            'cp' : 'C',
            'type' : type,
            'select_value' : select_values[0],
            'term' : 1,
            'volume' : 1,
            'iv_range' : [0, 999]
            }
call2 = {'entry_dates' : entry_dates,
            'table' : table,
            'cp' : 'C',
            'type' : type,
            'select_value' : select_values[1],
            'term' : 1,
            'volume' : -1,
            'iv_range' : [0, 999]
            }
call3 = {'entry_dates' : entry_dates,
            'table' : table,
            'cp' : 'C',
            'type' : type,
            'select_value' : select_values[2],
            'term' : 1,
            'volume' : -1,
            'iv_range' : [0, 999]
            }

queried = bt.backtest(call1, call2, call3)

# 변수
stop_dates = [[]]
dte_stop = 1
profit = [0.1, 0.25, 0.5, 1, 2, 3, 4]
loss = [-2, -1.5, -1, -0.5, -0.25, -0.1]
is_complex_pnl = True
is_intraday_stop = False
start_date = '20100101'
end_date = datetime.datetime.today().strftime("%Y-%m-%d")
show_chart = False
use_polars = True

trade_vars = list(generate_iterables(stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop))

df = pd.DataFrame()

for stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop in trade_vars:

    name = f'{stop_dates}_{dte_stop}_{profit}_{loss}_{is_complex_pnl}'

    result = queried.equal_inout(stop_dates,
                                    dte_stop,
                                    profit,
                                    loss,
                                    is_complex_pnl,
                                    is_intraday_stop,
                                    start_date,
                                    end_date,
                                    show_chart,
                                    use_polars)
    res = metrics(result)
    res.index = [name]
    df = pd.concat([df, res], axis = 0)
    df = df.sort_values('calmar', ascending = False)


#%% test_best
var = dict(
stop_dates = [],
dte_stop = 1,
profit = 0.25,
loss = -2,
is_complex_pnl = True,
is_intraday_stop = False,
start_date = '20100101',
end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
show_chart = True,
use_polars = True
)

best = queried.equal_inout(**var)

# %%
