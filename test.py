#%%
import numpy as np
import pandas as pd
import backtest as bt
import pathlib
import get_entry_exit
import datetime
import ast
import typing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
test_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option_test.db")

df_k200 = bt.get_timeseries(db_path, "k200")['k200']

'''test functions '''

def split(strat):

# 1. entry
    entry = strat.partition("_table")[0]
    strat = strat.partition("_table")[2]

# 2. table
    table = strat.partition("_type")[0]
    table = ast.literal_eval(table)
    strat = strat.partition("_type")[2]

# 3. type / offset_type
    atype = strat.partition("_term")[0]
    type = atype.split("&")[0]
    offset_type = atype.split("&")[1]

    strat = strat.partition("_term")[2]

#4. term
    terms = strat.partition("_volume")[0]
    terms = ast.literal_eval(terms)
    strat = strat.partition("_volume")[2]

#5. volume
    volume = strat.partition("_ref")[0]
    volume = ast.literal_eval(volume)
    strat = strat.partition("_ref")[2]
    
#6. ref_value / offset_value
    if offset_type == 'delta':
        common_value = ast.literal_eval(strat.partition("_exit")[0])
        select_value = common_value[0]
        offset = common_value[1:]
        strat = strat.partition("_exit")[2]
    else:
        common_value = strat.partition("_exit")[0].split("_")
        select_value = ast.literal_eval(common_value[0])
        offset = ast.literal_eval(common_value[1].removeprefix("ofst"))
        strat = strat.partition("_exit")[2]
    
#6. trading variables
    vars = strat.split("_")

    exit = vars[0]
    dte_stop = int(vars[1].removeprefix("dte"))
    profit = float(vars[2].removeprefix("p"))
    loss = float(vars[3].removeprefix("l"))
    is_complex_strat = ast.literal_eval(vars[4])

    result = dict(
        entry = entry,
        table = table,
        type = type,
        offset_type = offset_type,
        terms = terms,
        volume = volume,
        select_value = select_value,
        offset = offset,
        exit = exit,
        dte_stop = dte_stop,
        profit = profit,
        loss = loss,
        is_complex_strat = is_complex_strat
    )

    return result

def split_both(strat):

# 1. entry
    entry = strat.partition("_table")[0]
    strat = strat.partition("_table")[2]

# 2. table
    table = strat.partition("_type")[0]
    table = ast.literal_eval(table)
    strat = strat.partition("_type")[2]

# 3. type / offset_type
    atype = strat.partition("_term")[0]
    type = atype.split("&")[0]
    offset_type = atype.split("&")[1]

    strat = strat.partition("_term")[2]

    i = 0
    sp = strat.split("_")

#4. term
    terms = sp.pop(0)
    terms = ast.literal_eval(terms)

    if offset_type == 'delta':
        call_volume = ast.literal_eval(sp[i].removeprefix("callvolume"))
        put_volume = ast.literal_eval(sp[i + 1].removeprefix("putvolume"))
        call_ref = ast.literal_eval(sp[i + 2].removeprefix("callref"))[0]
        call_offset = ast.literal_eval(sp[i + 2].removeprefix("callref"))[1:]
        put_ref = ast.literal_eval(sp[i + 3].removeprefix("putref"))[0]
        put_offset = ast.literal_eval(sp[i + 3].removeprefix("putref"))[1:]
        i -= 2
    else:
        call_volume = ast.literal_eval(sp[i].removeprefix("callvolume"))
        put_volume = ast.literal_eval(sp[i + 1].removeprefix("putvolume"))
        call_ref = float(sp[i + 2].removeprefix("callref"))
        call_offset = ast.literal_eval(sp[i + 3].removeprefix("callofst"))
        put_ref = float(sp[i + 4].removeprefix("putref"))
        put_offset = ast.literal_eval(sp[i + 5].removeprefix("putofst"))

    exit = sp[i + 6].removeprefix("exit")
    dte_stop = int(sp[i + 7].removeprefix("dte"))
    profit = float(sp[i + 8].removeprefix("p"))
    loss = float(sp[i + 9].removeprefix("l"))
    is_complex_strat = eval(sp[i + 10])

    result = dict(
        entry = entry,
        table = table,
        type = type,
        offset_type = offset_type,
        terms = terms,
        call_volume = call_volume,
        call_select_value = call_ref,
        call_offset = call_offset,
        put_volume = put_volume,
        put_select_value = put_ref,
        put_offset = put_offset,
        exit = exit,
        dte_stop = dte_stop,
        profit = profit,
        loss = loss,
        is_complex_strat = is_complex_strat
    )

    return result

def get_entry_date(strat, cp):

    if cp in ["C", "P"]:
        split_strat = split(strat)
        entry = split_strat['entry']
        table = split_strat['table']
        first_term = split_strat['terms'][0]
        first_table = split_strat['table'][0]

        try:
            second_term = split_strat['terms'][1]
            second_table = split_strat['table'][1]
        except:
            pass

    elif cp == "B":
        split_strat = split_both(strat)
        entry = split_strat['entry']
        table = split_strat['table']
        first_term = split_strat['terms'][0]
        first_table = split_strat['table'][0]

        try:
            second_term = split_strat['terms'][1]
            second_table = split_strat['table'][1]
        except:
            pass

    entry_dates = {
        "mon": lambda: df_k200.weekday(0),
        "tue": lambda: df_k200.weekday(1),
        "wed": lambda: df_k200.weekday(2),
        "thu": lambda: df_k200.weekday(3),
        "fri": lambda: df_k200.weekday(4),

        "monivbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', test_path)) ,
        "monivbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', test_path)) ,
        "tueivbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', test_path)) ,
        "tueivbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', test_path)) ,
        "wedivbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', test_path)) ,
        "wedivbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', test_path)) ,
        "thuivbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', test_path)) ,
        "thuivbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', test_path)) ,
        "friivbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', test_path)) ,
        "friivbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', test_path)) ,

        "monivabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', test_path)) ,
        "monivabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', test_path)) ,
        "tueivabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', test_path)) ,
        "tueivabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', test_path)) ,
        "wedivabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', test_path)) ,
        "wedivabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', test_path)) ,
        "thuivabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', test_path)) ,
        "thuivabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', test_path)) ,
        "friivabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', test_path)) ,
        "friivabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', test_path)) ,

        "monskewbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', test_path)) ,
        "monskewbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', test_path)) ,
        "tueskewbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', test_path)) ,
        "tueskewbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', test_path)) ,
        "wedskewbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', test_path)) ,
        "wedskewbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', test_path)) ,
        "thuskewbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', test_path)) ,
        "thuskewbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', test_path)) ,
        "friskewbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', test_path)) ,
        "friskewbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', test_path)) ,

        "monskewabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', test_path)) ,
        "monskewabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', test_path)) ,
        "tueskewabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', test_path)) ,
        "tueskewabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', test_path)) ,
        "wedskewabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', test_path)) ,
        "wedskewabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', test_path)) ,
        "thuskewabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', test_path)) ,
        "thuskewabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', test_path)) ,
        "friskewabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', test_path)) ,
        "friskewabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', test_path)) ,

        "moncalendarbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', test_path)) ,
        "moncalendarbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', test_path)) ,
        "tuecalendarbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', test_path)) ,
        "tuecalendarbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', test_path)) ,
        "wedcalendarbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', test_path)) ,
        "wedcalendarbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', test_path)) ,
        "thucalendarbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', test_path)) ,
        "thucalendarbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', test_path)) ,
        "fricalendarbelow50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', test_path)) ,
        "fricalendarbelow25": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', test_path)) ,
                            
        "moncalendarabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', test_path)) ,
        "moncalendarabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(0), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', test_path)) ,
        "tuecalendarabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', test_path)) ,
        "tuecalendarabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(1), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', test_path)) ,
        "wedcalendarabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', test_path)) ,
        "wedcalendarabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(2), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', test_path)) ,
        "thucalendarabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', test_path)) ,
        "thucalendarabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(3), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', test_path)) ,
        "fricalendarabove50": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', test_path)) ,
        "fricalendarabove75": lambda: get_entry_exit.get_date_intersect(df_k200.weekday(4), 
                            get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', test_path)) ,


        "psartrendup": lambda: df_k200.psar.trend('l'), # psar 상승하고있는 경우
        "psartrenddown": lambda: df_k200.psar.trend('s'),
        "stochturnup": lambda: df_k200.stoch.rebound1('l', 10, 3, 3),
        "stochturndown": lambda: df_k200.stoch.rebound1('s', 10, 3, 3),
        "rsiturnup": lambda: df_k200.rsi.rebound('l', 14, 100, 30, 60),
        "rsiturndown": lambda: df_k200.rsi.rebound('s', 14, 100, 30, 60),
        "stochturndown": lambda: df_k200.stoch.rebound1('s', 10, 3, 3),
        "pricerebound" : lambda : df_k200.priceaction.change_recent(change = -0.03),
        "pricedrop" : lambda : df_k200.priceaction.change_recent(change = 0.03)
    }

    return entry_dates.get(entry)()

def get_exit_date(strat, cp):

    if cp in ["C", "P"]:
        split_strat = split(strat)
        exit = split_strat['exit']
        table = split_strat['table']
        first_term = split_strat['terms'][0]
        first_table = split_strat['table'][0]

    elif cp == "B":
        split_strat = split_both(strat)
        exit = split_strat['exit']
        table = split_strat['table']
        first_term = split_strat['terms'][0]
        first_table = split_strat['table'][0]


    exit_dates = {
        "noexit": lambda: []
    }

    return exit_dates.get(exit)()

def exit(res):

    entry_reset = res['entry'].reset_index(level = 'date')
    exit_reset = res['exit'].reset_index(level = 'date')
    joined = exit_reset.merge(entry_reset['k200'], left_index = True, right_index = True, how = 'inner')
    joined = joined.set_index(['date', joined.index])

    return joined

def find(res, entry_date):
    df_check = res['check']
    res = df_check.loc[(slice(None), entry_date), :]
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

def combine_pnl(*strategies):

    # 1) 기준이 되는 df_k200
    df_k200 = bt.get_timeseries(db_path, "k200")['k200'].loc['2010-01-01' :][['close']]
    df = df_k200    
    i = 0
    for result in strategies:
        
        pnl = runtest.execute(*result)['pnl']['daily_pnl']
        pnl.name = result[0]
        df = pd.merge(df, pnl, how = 'left', left_index = True, right_index = True)
        i += 1
        print(result)

    return df

def profile(res):

    res_dict = dict(
    daily_max_loss = res['daily_pnl'].min(),
    daily_max_gain = res['daily_pnl'].max(),
    mdd = res['dd'].min(),
    calmar = res['cum_pnl'].iloc[-1] / ((res.index[-1] - res.index[0]).days / 365) / abs(res['dd'].min())
    )

    return res_dict

# (weekly 전용) 위클에서 특정일 주물의 특정일에 해당하는 counterparty 주물의 전략 반환
def get_counter_date(strat):

    split = strat.partition("_table")
    
    entry_condition = split[0]
    old_entry = entry_condition[0:3]
    split_rest = split[2].partition("_type")

    old_table = split_rest[0]
    table = ast.literal_eval(old_table)
    first_table = table[0]

    if first_table == 'weekly_mon':
        counter = {
                'mon' : 'thu',
                'tue' : 'fri',
                'wed' : 'mon',
                'thu' : 'tue',
                'fri' : 'wed'}

    elif first_table == 'weekly_thu':
        counter = {
                'thu' : 'mon',
                'fri' : 'tue',
                'mon' : 'wed',
                'tue' : 'thu',
                'wed' : 'fri'}
        
    new_entry = counter[old_entry]
    new_table = list(map(lambda x : "weekly_mon" if x == 'weekly_thu' else 'weekly_thu', table))

    counter_strat = strat.replace(old_entry, new_entry, 1)
    counter_strat = counter_strat.replace(old_table, str(tuple(new_table)), 1)

    return counter_strat

# 주물에서 특정일 주물의 특정일에 해당하는 counterparty 주물의 동일 dte 특정일에 대한 전략 text 생성하는 용도
def get_counter_df(strat_string):

    df = pd.read_excel(f"C:/Users/kwan/Desktop/전략/{strat_string}.xlsx", header = 0, index_col = 0)
    
    new_index = df.index.map(get_counter_date)
    
    df_counter = df.reindex(new_index)

    df_counter.to_csv(f"C:/Users/kwan/Desktop/전략/{strat_string}_copy.csv", encoding = 'cp949')

    return df_counter

# 월물에서 특정 요일 찝어서 진입하는 경우 다른 날짜에도 충분히 유효한지 체크용
def another_dates(strat, strat_function, cp = "C"):

    strat_1 = strat.replace("mon", "tue", 1)
    strat_2 = strat.replace("mon", "wed", 1)
    strat_3 = strat.replace("mon", "thu", 1)
    strat_4 = strat.replace("mon", "fri", 1)

    b = strat_function(strat_1,
        cp,
        1)
    print(b['res']['premium'].mean())
    print(b['res']['premium'].median())

    c = strat_function(strat_2,
        cp,
        1)
    print(c['res']['premium'].mean())
    print(c['res']['premium'].median())

    d = strat_function(strat_3,
        cp,
        1)
    print(d['res']['premium'].mean())
    print(d['res']['premium'].median())

    e = strat_function(strat_4,
        cp,
        1)
    print(e['res']['premium'].mean())
    print(e['res']['premium'].median())

    res = pd.concat(map(metrics, [b, c, d, e]), axis = 0, join = 'inner', ignore_index = True)
    res.index = pd.Index(['tue', ' wed', 'thu', 'fri'])

    return b, c, d, e, res


    new_res = df['res'].loc[df['res']['days_taken'] > days_taken]
    n_total = len(df['res'])
    n_data = len(df['res']) - len(new_res)

    new_check = df['check'].loc[(slice(None), new_res.index.get_level_values('entry_date')), :]

    new_pnl = new_check['daily_pnl'].groupby('date').sum()
    new_pnl = new_pnl.reindex(df['pnl'].index).fillna(0).to_frame()
    new_pnl['cum_pnl'] = new_pnl['daily_pnl'].cumsum()
    new_pnl['dd'] = new_pnl['cum_pnl'] - new_pnl['cum_pnl'].cummax()

    new_pnl.plot()

    return dict(res = new_res,
                check = new_check,
                pnl = new_pnl,
                n_total = n_total,
                n_removed = n_data,
                cum_pnl = new_pnl['cum_pnl'].iloc[-1])

class runtest:
    @classmethod
    def _parse_strategy(cls, strat):
        """Parse strategy string and determine its type"""
        if '_callvolume' in strat and '_putvolume' in strat:
            # Both call and put strategy
            strat_split = split_both(strat)
            # Check if it's a conversion or strangle based on volume signs
            # Handle both tuple and list cases
            nearest_call = strat_split['call_volume'][0] if isinstance(strat_split['call_volume'], (tuple, list)) else strat_split['call_volume']
            nearest_put = strat_split['put_volume'][0] if isinstance(strat_split['put_volume'], (tuple, list)) else strat_split['put_volume']
            
            # Check terms length first
            if len(strat_split['terms']) == 2:
                if nearest_call * nearest_put < 0:
                    return 'conversion'
                    # 콜 풋 부호 다름 
                elif nearest_call * nearest_put > 0:
                    return 'condor'
                    # 콜 풋 부호 같음
            else:
                return 'strangle'
        else:
            # Single direction strategy
            strat_split = split(strat)
            if len(strat_split['terms']) == 1:
                return 'oneleg'
            elif len(strat_split['terms']) == 2:
                return 'twoleg'
            elif len(strat_split['terms']) == 3:
                return 'threeleg'
        return None

    @classmethod
    def _oneleg(cls, strat, cp="C", n=1):
        """Execute one-leg strategy"""
        entry_date = get_entry_date(strat, cp)
        exit_date = get_exit_date(strat, cp)
        strat_split = split(strat)

        leg1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': cp,
            'type': strat_split['type'],
            'select_value': strat_split['select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }

        queried = bt.backtest(leg1, option_path=test_path)

        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )

        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n

        return result

    @classmethod
    def _twoleg(cls, strat, cp="C", n=1):
        """Execute two-leg strategy"""
        entry_date = get_entry_date(strat, cp)
        exit_date = get_exit_date(strat, cp)
        strat_split = split(strat)

        leg1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': cp,
            'type': strat_split['type'],
            'select_value': strat_split['select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }

        if strat_split['offset_type'] == 'delta':
            leg2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['offset'][0],
                'term': strat_split['terms'][1],
                'volume': strat_split['volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999]
            }
        else:
            leg2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['select_value'],
                'term': strat_split['terms'][0],
                'volume': strat_split['volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999],
                'offset': strat_split['offset'][0],
                'term_offset': strat_split['terms'][1]
            }

        queried = bt.backtest(leg1, leg2, option_path=test_path)

        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )

        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n

        return result

    @classmethod
    def _threeleg(cls, strat, cp="C", n=1):
        """Execute three-leg strategy"""
        entry_date = get_entry_date(strat, cp)
        exit_date = get_exit_date(strat, cp)
        strat_split = split(strat)

        leg1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': cp,
            'type': strat_split['type'],
            'select_value': strat_split['select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }

        if strat_split['offset_type'] == 'delta':
            leg2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['offset'][0],
                'term': strat_split['terms'][1],
                'volume': strat_split['volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999]
            }
            leg3 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][2],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['offset'][1],
                'term': strat_split['terms'][2],
                'volume': strat_split['volume'][2],
                'dte': [1, 999],
                'iv_range': [0, 999]
            }
        else:
            leg2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['select_value'],
                'term': strat_split['terms'][0],
                'volume': strat_split['volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999],
                'offset': strat_split['offset'][0],
                'term_offset': strat_split['terms'][1]
            }
            leg3 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][2],
                'cp': cp,
                'type': strat_split['type'],
                'select_value': strat_split['select_value'],
                'term': strat_split['terms'][0],
                'volume': strat_split['volume'][2],
                'dte': [1, 999],
                'iv_range': [0, 999],
                'offset': strat_split['offset'][1],
                'term_offset': strat_split['terms'][2]
            }

        queried = bt.backtest(leg1, leg2, leg3, option_path=test_path)

        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )

        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n

        return result

    @classmethod
    def _strangle(cls, strat, cp="B", n=1):
        """Execute strangle strategy"""
        entry_date = get_entry_date(strat, "B")
        exit_date = get_exit_date(strat, "B")
        strat_split = split_both(strat)

        call1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'C',
            'type': strat_split['type'],
            'select_value': strat_split['call_select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['call_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        put1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'P',
            'type': strat_split['type'],
            'select_value': strat_split['put_select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['put_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        queried = bt.backtest(call1, put1, option_path=test_path)
        
        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )
        
        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n
        
        return result

    @classmethod
    def _conversion(cls, strat, cp="B", n=1):
        """Execute conversion strategy
        condor 이랑 다른 점 : 1) callvolume / putvolume 부호가 다름 (condor 랑 구별 기준)
        terms 가 콜이랑 풋이랑 각자 반영함 -> 콜풋 양쪽 전략중에서는 예외조건
        """
        entry_date = get_entry_date(strat, "B")
        exit_date = get_exit_date(strat, "B")
        strat_split = split_both(strat)

        call1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'C',
            'type': strat_split['type'],
            'select_value': strat_split['call_select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['call_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        put1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'P',
            'type': strat_split['type'],
            'select_value': strat_split['put_select_value'],
            'term': strat_split['terms'][1],
            'volume': strat_split['put_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        queried = bt.backtest(call1, put1, option_path=test_path)
        
        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )
        
        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n
        
        return result

    @classmethod
    def _condor(cls, strat, cp="B", n=1):
        """Execute condor strategy"""
        entry_date = get_entry_date(strat, "B")
        exit_date = get_exit_date(strat, "B")
        strat_split = split_both(strat)

        call1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'C',
            'type': strat_split['type'],
            'select_value': strat_split['call_select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['call_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        put1 = {
            'entry_dates': entry_date,
            'table': strat_split['table'][0],
            'cp': 'P',
            'type': strat_split['type'],
            'select_value': strat_split['put_select_value'],
            'term': strat_split['terms'][0],
            'volume': strat_split['put_volume'][0],
            'dte': [1, 999],
            'iv_range': [0, 999]
        }
        
        if strat_split['offset_type'] == 'delta':
            call2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': 'C',
                'type': strat_split['type'],
                'select_value': strat_split['call_offset'][0],
                'term': strat_split['terms'][1],
                'volume': strat_split['call_volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999]
            }
            
            put2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': 'P',
                'type': strat_split['type'],
                'select_value': strat_split['put_offset'][0],
                'term': strat_split['terms'][1],
                'volume': strat_split['put_volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999]
            }
        else:
            call2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': 'C',
                'type': strat_split['type'],
                'select_value': strat_split['call_select_value'],
                'term': strat_split['terms'][0],
                'volume': strat_split['call_volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999],
                'offset': strat_split['call_offset'][0],
                'term_offset': strat_split['terms'][1]
            }
            
            put2 = {
                'entry_dates': entry_date,
                'table': strat_split['table'][1],
                'cp': 'P',
                'type': strat_split['type'],
                'select_value': strat_split['put_select_value'],
                'term': strat_split['terms'][0],
                'volume': strat_split['put_volume'][1],
                'dte': [1, 999],
                'iv_range': [0, 999],
                'offset': strat_split['put_offset'][0],
                'term_offset': strat_split['terms'][1]
            }
        
        queried = bt.backtest(call1, call2, put1, put2, option_path=test_path)
        
        var = dict(
            stop_dates=exit_date,
            dte_stop=strat_split['dte_stop'],
            profit=strat_split['profit'],
            loss=strat_split['loss'],
            is_complex_pnl=strat_split['is_complex_strat'],
            is_intraday_stop=False,
            start_date='20100101',
            end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            show_chart=True,
            use_polars=True
        )
        
        result = queried.equal_inout(**var)
        result['pnl'] = result['pnl'] * n
        
        return result

    @classmethod
    def execute(cls, strat, cp=None, n=1):
        """
        Unified interface for executing any strategy
        
        Parameters:
        -----------
        strat : str
            Strategy string
        cp : str, optional
            Call/Put indicator. If None, will be determined from strategy string
        n : int, optional
            Multiplier for the strategy (default=1)
            
        Returns:
        --------
        dict
            Result dictionary containing pnl, check, entry, exit, and res dataframes
        """
        # Determine strategy type
        strategy_type = cls._parse_strategy(strat)
        
        # Set default cp if not provided
        if cp is None:
            cp = "B" if strategy_type in ['strangle', 'conversion', 'condor'] else "C"
        
        # Execute appropriate strategy
        if strategy_type == 'strangle':
            return cls._strangle(strat, cp, n)
        elif strategy_type == 'conversion':
            return cls._conversion(strat, cp, n)
        elif strategy_type == 'condor':
            return cls._condor(strat, cp, n)
        elif strategy_type == 'oneleg':
            return cls._oneleg(strat, cp, n)
        elif strategy_type == 'twoleg':
            return cls._twoleg(strat, cp, n)
        elif strategy_type == 'threeleg':
            return cls._threeleg(strat, cp, n)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

# Usage example:
# executor = StrategyExecutor()
# result = executor.execute(strat)  # strat can be any valid strategy string

#%% 개별전략 테스트

strat = "thu_table('weekly_thu', 'weekly_mon')_typedelta&moneyness_term(1, 1)_volume(2, -3)_ref-0.5_ofst[5.0]_exitnoexit_dte1_p0.5_l-0.5_True"
cp = "P"

try: 
    table = split(strat)['table']
    first_term = split(strat)['terms'][0]
    first_table = split(strat)['table'][0]
except:
    table = split_both(strat)['table']
    first_term = split_both(strat)['terms'][0]
    first_table = split_both(strat)['table'][0]

# 반영한 필터 적용 로직
# 1) 0dte 마감exit -> 내재가치만 남으므로 전부 킵해놓고 나머지 ->
#   2) 1번에서 0dte 가 아닌 애들 중에서, entry 시 IV 필터 적용
# 3) exit 시점에서 ITM인 경우 -> IV 왜곡 심하므로 전부 킵 (True) 해놓고 나머지
# 4) exit 시점에서 보수적인 접근 목적으로 losses 는 보수적으로 전부 킵 (True) 해놓고 나머지
#   5) 3/4번에서 exit 시 ITM 이거나, loss 로 끝나지 않는(=profit으로 끝난) 애들 중에서 IV 필터 적용

def iv_check(res, cp, lower= 0.95, upper = 1.2, keep_itm = True, how_deep = 2.5, keep_losses = True):

    ''' 여러 단계적 필터 조건에 따라 필터링되는 애들 빼고 나머지 애들의 "entry_date" 을 추출해서
    res 전체 대비 필터링된 entry만 다시 추리는 함수
    '''

    if cp == 'C':
        multiplier = -1
    else:
        multiplier = 1

    entry_numbers = res['exit'].shape[0]
    leg_numbers = res['exit'].shape[1]

    # 필터 1) dte = 0에 끝나는 일부 전략의 IV랑 하등 무관하게 내재가치로 계산하므로 걔들은 그냥 보유
    zerodte_idx = res['exit'].loc[res['exit']['min_dte'] == 0].index.get_level_values("entry_date")
    exit_non0dte = res['exit'].loc[res['exit']['min_dte'] != 0].sort_values("cum_pnl", ascending = False)
    idx = set(zerodte_idx)

    if not exit_non0dte.empty:

        idx_non0dte = exit_non0dte.index.get_level_values(level = 'entry_date')
        exit_non0dte = res['exit'].loc[(slice(None), idx_non0dte), :]
        entry_non0dte = res['entry'].loc[(slice(None), idx_non0dte), :]

        entry_non0dte = pd.concat([entry_non0dte.filter(regex = 'strike$'), entry_non0dte.filter(regex = 'iv$'), entry_non0dte[['daily_pnl', 'cum_pnl', 'k200']]], axis = 1) 
        exit_non0dte = pd.concat([exit_non0dte.filter(regex = 'strike$'), exit_non0dte.filter(regex = 'iv$'), exit_non0dte[['daily_pnl', 'cum_pnl', 'k200']]], axis = 1)

        # 기본 필터링 컨셉
        # 1) 유리한거만 빼고 불리한건 그냥 두기
        # 2) IV 왜곡이 손익에 큰 영향이 없는 경우는 그냥 두기
        #  2-1) ITM 인 경우 필터링 안 하기 : 손익이 IV와 크게 상관 없어짐. IV에 따라 정도는 다르나 이익/또는 손실난 포지션은 맞음

        if leg_numbers == 14:
            # 필터 2) : 진입할때 유리한거 빼기
            entry_idx = set()
            entry_cond = (entry_non0dte.filter(regex = "iv$") / entry_non0dte.filter(regex = "iv$").shift(axis = 1)).dropna(axis = 1)
            if not entry_cond.empty:
                entry_cond.columns = range(1, len(entry_cond.columns) + 1)
                cond1 = entry_cond[1] < upper # 진입할때 가운데 매도가 너무 비싸면 왜곡
                entry_idx = set(entry_cond.loc[cond1].index.get_level_values('entry_date'))

                # 필터 3) : 일정수준 이상 ITM인 애들은 IV 왜곡 가능성 높으므로 손익 그대로 두기
                itms_idx, losses_idx, exit_idx = set(), set(), set()
                itms_cond = multiplier * (exit_non0dte.iloc[:, 0] - exit_non0dte['k200']) >= how_deep

                if keep_itm:
                    itms_idx = set(exit_non0dte.loc[itms_cond].index.get_level_values('entry_date')) # 1. exit 할때 일정 수준 이상 내가격은 그냥 최종 계산시에 포함 / 내가격 아닌애들만 iv filtering 대상으로
                    non_itms = exit_non0dte.loc[~itms_cond]
                    if non_itms.empty:
                        idx = itms_idx.intersection(entry_idx).union(zerodte_idx)
                else:
                    non_itms = exit_non0dte

                if keep_losses:
                    losses_idx = set(non_itms.loc[non_itms['cum_pnl'] < 0].index.get_level_values('entry_date')) # 2. exit 할때 보수적으로 loss 는 그냥 포함 / profit 만 iv filtering 대상으로
                    profit_only = non_itms.loc[non_itms['cum_pnl'] >= 0]
                    if profit_only.empty:
                        idx = itms_idx.union(losses_idx).intersection(entry_idx).union(zerodte_idx)
                else:
                    profit_only = non_itms

                exit_cond = (profit_only.filter(regex = "iv$") / profit_only.filter(regex = "iv$").shift(axis = 1)).dropna(axis = 1)
                if not exit_cond.empty:
                    
                    exit_cond.columns = range(1, len(exit_cond.columns) + 1)
                    cond1 = exit_cond[1] > lower # 청산할때 매도한게 앞에 매수한거보다 너무 싸게 환매수하면 왜곡
                    exit_idx = set(exit_cond.loc[cond1].index.get_level_values('entry_date'))
                    idx = itms_idx.union(losses_idx).union(exit_idx).intersection(entry_idx).union(zerodte_idx)

        elif leg_numbers == 18: 
            # 필터 2) : 진입할때 유리한거 빼기
            entry_idx = set()
            entry_cond = (entry_non0dte.filter(regex = "iv$") / entry_non0dte.filter(regex = "iv$").shift(axis = 1)).dropna(axis = 1)
            if not entry_cond.empty:
                entry_cond.columns = range(1, len(entry_cond.columns) + 1)
                cond1 = entry_cond[1] < upper # 진입할때 가운데 매도가 너무 비싸면 왜곡
                cond2 = entry_cond[2] > lower # 진입할때 극외가 매수포가 너무 싸면 가격 왜곡

                entry_idx = entry_cond.loc[cond1 & cond2].index.get_level_values('entry_date')

                itms_idx, losses_idx, exit_idx = set(), set(), set()
                
                # 필터 3) : 일정수준 이상 ITM인 애들은 IV 왜곡 가능성 높으므로 손익 그대로 두기
                itms_cond = multiplier * (exit_non0dte.iloc[:, 0] - exit_non0dte['k200']) >= how_deep
                if keep_itm:
                    itms_idx = set(exit_non0dte.loc[itms_cond].index.get_level_values('entry_date')) # 1. exit 할때 일정 수준 이상 내가격은 그냥 최종 계산시에 포함 / 내가격 아닌애들만 iv filtering 대상으로
                    non_itms = exit_non0dte.loc[~itms_cond]
                    if non_itms.empty:
                        idx = itms_idx.intersection(entry_idx).union(zerodte_idx)
                else:
                    non_itms = exit_non0dte

                if keep_losses:
                    losses_idx = set(non_itms.loc[non_itms['cum_pnl'] < 0].index.get_level_values('entry_date')) # 2. exit 할때 보수적으로 loss 는 그냥 포함 / profit 만 iv filtering 대상으로
                    profit_only = non_itms.loc[non_itms['cum_pnl'] >= 0]
                    if profit_only.empty:
                        idx = itms_idx.union(losses_idx).intersection(entry_idx).union(zerodte_idx)
                else:
                    profit_only = non_itms

                exit_cond = (profit_only.filter(regex = "iv$") / profit_only.filter(regex = "iv$").shift(axis = 1)).dropna(axis = 1)
                if not exit_cond.empty:
                    
                    exit_cond.columns = range(1, len(exit_cond.columns) + 1)
                    cond1 = exit_cond[1] > lower # 청산할때 매도한게 앞에 매수한거보다 너무 싸게 환매수하면 왜곡
                    cond2 = exit_cond[2] < upper # 청산할때 극외가 매수포가 너무 비싸면 가격 왜곡
                    exit_idx = set(exit_cond.loc[cond1 & cond2].index.get_level_values('entry_date'))

                    idx = itms_idx.union(losses_idx).union(exit_idx).intersection(entry_idx).union(zerodte_idx)
 
    filtered_numbers = len(idx)
    filtered_pnl = res['check'].loc[(slice(None), list(idx)), :].sort_index()['daily_pnl'].cumsum()
    filtered_mdd = (filtered_pnl - filtered_pnl.cummax()).min()
    
    print(f"-------------------------\n"
          f"filtered_numbers: {filtered_numbers} / {len(res['exit'])} \n"
          f"filtered_pnl: {filtered_pnl.iloc[-1] if not filtered_pnl.empty else 0} \n"
          f"filtered_mdd: {filtered_mdd if not filtered_pnl.empty else 0} \n"
          f"---------------------")
    
    return zerodte_idx, entry_idx if 'entry_idx' in locals() else set(), exit_idx if 'exit_idx' in locals() else set(), res['exit'].loc[(slice(None), list(idx)), :] if not res['exit'].empty else pd.DataFrame()

lower = 0.95
upper = 1.2
how_deep = 3

def drop_conditions(res):
    ## 그때그때 수정
    cond1 = (res['entry']['min_dte'] > 7).droplevel('date') # 1) 월물이라면서 위클리는 빼기
    cond2 = (res['entry']['value_sum'] > 0).droplevel('date') # 2) 애초에 진입할때 net credit으로 진입한건 빼기
    # cond3 = (res['exit']['min_dte'] < 8).droplevel('date')
    entry_date = res['entry'].index.get_level_values('entry_date')[cond1]
    
    check = res['check'].loc[(slice(None), entry_date), :]
    entry = res['entry'].loc[(slice(None), entry_date), :]
    exit = res['exit'].loc[(slice(None), entry_date), :]
    pnl_df = check['daily_pnl'].droplevel('entry_date').groupby('date').sum().reindex(res['pnl'].index).fillna(0)
    pnl = pnl_df.cumsum()
    profit = pnl.iloc[-1]
    mdd = pnl - pnl.cummax()
    pnl.plot()

    return dict(check = check,
                entry = entry,
                exit = exit,
                pnl = pnl,
                profit = profit,
                mdd = mdd)

a = runtest.execute(strat, cp)
acheck = iv_check(a, cp, lower, upper, keep_itm = True, how_deep = how_deep, keep_losses = True) # ITM 은 전부 그대로 유지 / Loss는 전부 그대로 유지한 채 Profit (좋은거) 만 필터링 -> 제일 reasonable / 적당히 보수적
acheck1 = iv_check(a, cp, lower, upper, keep_itm = True, how_deep = how_deep, keep_losses = False) # ITM 은 전부 그대로 유지 / 나머지들 Profit/Loss 상관없이 필터링 -> 손실 과소평가되는 경향
acheck2 = iv_check(a, cp, lower, upper, keep_itm = False, how_deep = how_deep, keep_losses = True) # ITM/OTM 유무 상관없이 / Loss 는 전부 유지한 채 Profit 에서만 필터링 -> 반대로 너무 보수적
acheck3 = iv_check(a, cp, lower, upper, keep_itm = False, how_deep = how_deep, keep_losses = False) # 아무런 구분 없이 변동성 이상한 놈 죄다 필터링 -> ITM 들어가면 IV 왜곡되는 현상 무시하는 처사

if 'weekly' in first_table:
    aa = runtest.execute(get_counter_date(strat), cp, 1)
    aacheck = iv_check(aa, cp, lower, upper, keep_itm = True, how_deep = how_deep, keep_losses = True)
    aacheck1 = iv_check(aa, cp, lower, upper, keep_itm = True, how_deep = how_deep, keep_losses = False)
    aacheck2 = iv_check(aa, cp, lower, upper, keep_itm = False, how_deep = how_deep, keep_losses = True)
    aacheck3 = iv_check(aa, cp, lower, upper, keep_itm = False, how_deep = how_deep, keep_losses = False)

elif first_table == 'monthly':
    adate = another_dates(strat, runtest.execute, cp)
    mon = drop_conditions(a)
    tue = drop_conditions(adate[0])
    wed = drop_conditions(adate[1])
    thu = drop_conditions(adate[2])
    fri = drop_conditions(adate[3])
    print(adate[4])

print(f"mean premium : {a['res']['premium'].mean()}")
print(f"median premium : {a['res']['premium'].median()}")
print(f"premium std : {a['res']['premium'].std()}")
# sanity check

#1. (매수전략의 경우) entry 프리미엄이 0보다는 커야
# print("a_premium_above_0 :" + str(a['res'].loc[a['res']['premium'] > 0]['cum_pnl'].sum()))
# print("aflip_premium_above_0 :" + str(aflip['res'].loc[aflip['res']['premium'] > 0]['cum_pnl'].sum()))

#2. 월물매매하는경우 위클리는 빼고


#%%
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
from pathlib import Path

# 1. Load and group by 전략분류
df = pd.read_csv(pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "first_filter.txt"), sep="\t")
grouped = df.groupby("전략분류")

# 2. 결과 저장할 경로
Path("cluster_outputs").mkdir(exist_ok=True)

for name, group in grouped:
    print(f"Processing 전략분류: {name}")

    strategies = list(zip(group["strat"], group["cp"]))
    pnl_df = combine_pnl(*strategies).drop(columns = ['close'])  # 전략별 daily_pnl: shape (dates, strategies)

    if pnl_df.shape[1] < 3:
        print(f" → 전략 수 부족으로 스킵")
        continue

    pnl_df = pnl_df.fillna(0)  # 혹시 모를 NaN 제거

    # (A) PCA + KMeans
    pca = PCA(n_components=min(5, pnl_df.shape[1]))
    X = pca.fit_transform(pnl_df.T)
    kmeans = KMeans(n_clusters=min(3, pnl_df.shape[1]), random_state=0).fit(X)
    pca_clusters = kmeans.labels_

    # (B) 상관계수 기반 linkage clustering
    corr = pnl_df.corr().fillna(0)
    dist_corr = 1 - corr.abs()
    Z_corr = linkage(squareform(dist_corr.values), method="average")
    corr_clusters = fcluster(Z_corr, t=0.3, criterion='distance')  # corr > 0.7

    # (C) Wasserstein distance 기반 linkage clustering
    # 1. z-score normalize
    norm_df = pnl_df.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x, axis=0)

    # 2. Wasserstein distance matrix
    wasser_dist = np.array([
        [wasserstein_distance(norm_df[c1], norm_df[c2]) for c2 in norm_df.columns]
        for c1 in norm_df.columns
    ])

    # 3. linkage (주의: squareform 안씀)
    Z_wass = linkage(wasser_dist, method="average")
    wass_clusters = fcluster(Z_wass, t=0.6, criterion='distance')

    # 4. 저장용 DF 생성
    df_columns = pd.MultiIndex.from_arrays([pnl_df.columns, corr_clusters, pca_clusters])
    pnl_df.columns = df_columns
    pnl_df.columns.names = ['strat', 'corr', 'pca']
    pnl_df = pnl_df.cumsum()
    pnl_df = pnl_df.sort_index(axis = 1, level = 'corr')

    result = pd.DataFrame({
        "strat": group["strat"].values,
        "cp": group["cp"].values,
        "cluster_pca": pca_clusters,
        "cluster_corr": corr_clusters,
        "cluster_wasserstein": wass_clusters
    })
    
    out_path = Path(f"cluster_outputs/{name}.xlsx")

    with pd.ExcelWriter(path = out_path) as writer:
        result.to_excel(writer, sheet_name = "result", index = True)
        pnl_df.to_excel(writer, sheet_name = "cumret", index = True)

        print(f" → 저장 완료: {out_path}")


#%% 2차필터링 : 상관계수분석 및 재차 임의필터

# 전략 목록 불러오기
strategies = []

df = pd.read_csv(pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "second_filter.txt"), sep="\t")
strategies = list(zip(df["strat"], df["cp"]))

# 일간 손익 데이터 합치기
pnl_df = combine_pnl(*strategies).drop(columns=["close"])
pnl_df = pnl_df.fillna(0)

# (A) PCA + KMeans 클러스터링
pca = PCA(n_components=min(5, pnl_df.shape[1]))
X = pca.fit_transform(pnl_df.T)
kmeans = KMeans(n_clusters=min(3, pnl_df.shape[1]), random_state=0).fit(X)
pca_clusters = kmeans.labels_

# (B) 상관계수 기반 linkage clustering
corr = pnl_df.corr().fillna(0)
dist_corr = 1 - corr.abs()
Z_corr = linkage(squareform(dist_corr.values), method="average")
corr_clusters = fcluster(Z_corr, t=0.3, criterion='distance')

# MultiIndex 생성
df_columns = pd.MultiIndex.from_arrays([
    pnl_df.columns,
    corr_clusters,
    pca_clusters
], names=["strat", "corr", "pca"])

# 누적 수익률
pnl_df.columns = df_columns
cumret_df = pnl_df.cumsum()
cumret_df = cumret_df.sort_index(axis=1, level='corr')

# 클러스터링 결과 요약 테이블
result = pd.DataFrame({
    "strat": [s[0] for s in strategies],
    "cp": [s[1] for s in strategies],
    "cluster_pca": pca_clusters,
    "cluster_corr": corr_clusters
})

# 출력 경로
output_path = Path("res_second.xlsx")
fig1_path = Path("correlation_matrix.png")
fig2_path = Path("cumret_by_cluster.png")

# Excel 저장
with pd.ExcelWriter(output_path) as writer:
    result.to_excel(writer, sheet_name="result", index=False)
    cumret_df.to_excel(writer, sheet_name="cumret")
    corr.to_excel(writer, sheet_name="correlation_matrix")

# 히트맵 저장
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix (Second Filter)")
plt.tight_layout()
plt.savefig(fig1_path)
plt.close()
print(f"📈 Correlation matrix saved to: {fig1_path}")

# 클러스터별 누적 수익률 시각화
plt.figure(figsize=(14, 8))
unique_clusters = np.unique(corr_clusters)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

for i, strat in enumerate(pnl_df.columns.levels[0]):
    clust_id = result[result["strat"] == strat]["cluster_corr"].values[0]
    cumret_df[strat].plot(label=f"{strat} (C{clust_id})", color=colors[clust_id % len(colors)], alpha=0.6)

plt.title("Cumulative Returns by Cluster (Correlation Clustering)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.legend(loc="upper left", fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(fig2_path)
plt.close()
print(f"📊 Cumulative returns by cluster saved to: {fig2_path}")

print(f"✅ 모든 결과 저장 완료: {output_path}")

#%% 최적 전략 조합 찾기 (정수 단위)

strategies = []
df = pd.read_csv(pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "final.txt"), sep="\t")
strategies = list(zip(df["strat"], df["cp"]))

final_df = combine_pnl(*strategies)
# Exprot as csv
final_df.to_csv("final_df.csv", encoding = 'cp949')

#%% 다양한 포트폴리오 최적화 방법
import numpy as np
from scipy.optimize import differential_evolution
import time

def calculate_metrics(units, daily_pnl, selected_columns = None):
    # 정수 단위로 변환
    units = np.round(units).astype(int)
    
    # 가중치 계산 (단위 수 / 전체 단위 수)
    # weights = units / units.sum()
    weights = units

    if selected_columns is not None:
        daily_pnl = daily_pnl[selected_columns]
    
    # 가중치 적용된 일일 손익 계산
    weighted_daily_pnl = (daily_pnl * weights).sum(axis=1)
    
    # 누적 수익률 계산
    cum_pnl = weighted_daily_pnl.cumsum()
    
    # MDD 계산
    mdd = (cum_pnl - cum_pnl.cummax()).min()
    
    # 연간 수익 계산
    recent_years = weighted_daily_pnl.loc['2010-01-01':]
    annualize_factor = (weighted_daily_pnl.index[-1] - weighted_daily_pnl.index[0]).days / 365
    annual_gain = recent_years.sum() / annualize_factor
    
    # 일일 수익률의 표준편차
    daily_std = weighted_daily_pnl.std()
    
    # 하방 변동성 (음수 수익률만 고려)
    downside_std = weighted_daily_pnl[weighted_daily_pnl < 0].std()
    
    # Calmar Ratio
    calmar_ratio = annual_gain / abs(mdd)
    
    # Sharpe Ratio (무위험 수익률 = 0 가정)
    sharpe_ratio = (annual_gain / 252) / daily_std
    
    # Sortino Ratio
    sortino_ratio = (annual_gain / 252) / downside_std
    
    return {
        'calmar': calmar_ratio,
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'volatility': daily_std,
        'mdd': mdd,
        'annual_gain': annual_gain
    }

def optimize_portfolio(daily_pnl, objective='calmar', min_units_per_strategy = 0, max_units_per_strategy=10):
    n_strategies = daily_pnl.shape[1]
    
    # Cache correlation matrix for max_diversification objective
    corr_matrix = daily_pnl.corr() if objective == 'max_diversification' else None
    inv_corr = np.linalg.inv(corr_matrix) if corr_matrix is not None else None
    
    def objective_function(units):
        metrics = calculate_metrics(units, daily_pnl)
        if objective == 'calmar':
            return -metrics['calmar']  # 최소화 문제로 변환
        elif objective == 'sharpe':
            return -metrics['sharpe']
        elif objective == 'sortino':
            return -metrics['sortino']
        elif objective == 'min_vol':
            return metrics['volatility']
        elif objective == 'max_diversification':
            weights = units / units.sum()
            return -np.sqrt(weights @ inv_corr @ weights)
    
    bounds = [(min_units_per_strategy, max_units_per_strategy) for _ in range(n_strategies)]
    
    # 여러 번의 최적화 실행 (local minima/maxima 문제 해결)
    best_result = None
    best_value = float('inf')
    
    for _ in range(5):  # 5회에 걸쳐 서로 다른 random starting point 에서 시작해서 제일 좋은거 추리기
        start_time = time.time()
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            popsize=50,  # 20에서 50으로 증가
            mutation=(0.5, 1.0),
            recombination=0.7,
            maxiter=100,  # 50에서 100으로 증가
            seed=np.random.randint(0, 10000),  # 랜덤 seed 사용
            workers=1  # 병렬 처리 활성화
        )
        
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
        end_time = time.time()

        print(f"time taken for a single loop : {(end_time - start_time) / 60} minutes")
    
    optimal_units = np.round(best_result.x).astype(int)
    
    return optimal_units, daily_pnl.columns

daily_pnl = final_df.drop(columns=['close'])

# 다양한 목적함수로 최적화 실행
# objectives = ['calmar']
# 다양한 목적함수로 최적화 실행
objectives = ['calmar', 'sharpe', 'sortino']
results = {}

for obj in objectives:
    print(f"\nOptimizing for {obj}...")
    units, strategies = optimize_portfolio(daily_pnl, objective=obj, min_units_per_strategy = 0, max_units_per_strategy = 10)
    metric = calculate_metrics(units, daily_pnl)
    
    print(f"\nOptimal Strategy Units ({obj}):")
    for strat, unit in zip(strategies, units):
        print(f"{strat}: {unit} units")
    
    print(f"\nPortfolio Performance ({obj}):")
    print(f"Total Units: {units.sum()}")
    print(f"Annual Gain: {metric['annual_gain']:.2f}")
    print(f"Maximum Drawdown: {metric['mdd']:.2f}")
    print(f"Volatility: {metric['volatility']:.2f}")
    print(f"Calmar Ratio: {metric['calmar']:.2f}")
    print(f"Sharpe Ratio: {metric['sharpe']:.2f}")
    print(f"Sortino Ratio: {metric['sortino']:.2f}")
    
    # 결과 저장
    results[obj] = {
        'units': units,
        'strategies': strategies,
        'metric': metric
    }

# 결과 시각화
plt.figure(figsize=(15, 10))

# 각 최적화 방법별 누적 수익률 플롯
for obj in objectives:
    units = results[obj]['units']
    strategies = results[obj]['strategies']
    weights = units
    weighted_pnl = (daily_pnl[strategies] * weights).sum(axis=1)
    cum_pnl = weighted_pnl.cumsum()
    cum_pnl.plot(label=obj)

plt.title('Cumulative PnL Comparison of Different Optimization Methods')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.legend()
plt.grid(True)
plt.show()

# 성과 지표 비교
metrics_df = pd.DataFrame({
    obj: results[obj]['metric'] for obj in objectives
}).T

result_df = pd.DataFrame()
for i in results.keys():
    dummy = pd.DataFrame(results[i]['units'], index = results[i]['strategies'], columns = [str(i)])
    result_df = pd.concat([result_df, dummy], axis = 1)

result_df = result_df.T

print("\nPerformance Metrics Comparison:")
print(metrics_df)


#%%

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import TimeSeriesSplit

# Load the final_df.csv data
final_df = pd.read_csv('final_df.csv', index_col=0, parse_dates=True)

# Define the performance metrics
def calculate_metrics(weights, daily_pnl):
    weights = np.round(weights).astype(int)
    weighted_pnl = (daily_pnl * weights).sum(axis=1)
    cum_pnl = weighted_pnl.cumsum()
    mdd = (cum_pnl - cum_pnl.cummax()).min()
    annual_return = weighted_pnl.mean() * 252
    annual_vol = weighted_pnl.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    calmar = annual_return / abs(mdd) if mdd != 0 else 0
    sortino = annual_return / (weighted_pnl[weighted_pnl < 0].std() * np.sqrt(252)) if weighted_pnl[weighted_pnl < 0].std() != 0 else 0

    return -calmar  # Negative because we want to maximize in DE

# Define a walk-forward cross-validation optimization
def optimize_walkforward(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_weights = []

    for train_index, test_index in tscv.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        bounds = [(0, 10)] * train_df.shape[1]

        result = differential_evolution(
            calculate_metrics,
            bounds,
            args=(train_df,),
            strategy='best1bin',
            maxiter=500,
            popsize=10,
            tol=1e-6,
            seed=42
        )

        best_weights.append(result.x)

        print(f'Training on {train_df.index[0]} to {train_df.index[-1]}')
        print(f'Testing on {test_df.index[0]} to {test_df.index[-1]}')

    return best_weights

# Load the data
daily_pnl = final_df.drop(columns=['close']).fillna(0)

# Perform walk-forward optimization
optimal_weights = optimize_walkforward(daily_pnl)
print("Optimal Weights per Fold:", optimal_weights)


#%%

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from hmmlearn import hmm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the final_df.csv data
final_df = pd.read_csv('final_df.csv', index_col=0, parse_dates=True)

# Define the performance metrics
def calculate_metrics(weights, daily_pnl):
    weights = np.round(weights).astype(int)
    weighted_pnl = (daily_pnl * weights).sum(axis=1)
    cum_pnl = weighted_pnl.cumsum()
    mdd = (cum_pnl - cum_pnl.cummax()).min()
    annual_return = weighted_pnl.mean() * 252
    annual_vol = weighted_pnl.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    calmar = annual_return / abs(mdd) if mdd != 0 else 0
    sortino = annual_return / (weighted_pnl[weighted_pnl < 0].std() * np.sqrt(252)) if weighted_pnl[weighted_pnl < 0].std() != 0 else 0

    return -calmar  # Negative because we want to maximize in DE

# Regime Detection using Hidden Markov Model (HMM)
def detect_regimes(df, n_states=3):
    returns = df.pct_change().dropna()
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000)
    model.fit(returns.values)
    regimes = model.predict(returns.values)
    df['regime'] = regimes
    return df
