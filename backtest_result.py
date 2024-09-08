#%%

import pandas as pd
import numpy as np
import option_calc as calc
import compute
from datetime import datetime
import backtest

df_monthly = pd.read_pickle("./working_data/df_monthly.pkl")
df_weekly = pd.read_pickle("./working_data/df_weekly.pkl")

k200 = pd.read_pickle('./working_data/df_k200.pkl')
vkospi = pd.read_pickle('./working_data/df_vkospi.pkl')
vix = pd.read_pickle('./working_data/df_vix.pkl')

data_from = '2010-01' # 옛날에는 행사가가 별로 없어서 전략이 이상하게 나감

df_monthly = df_monthly.sort_index().loc[data_from:]
df_weekly = df_weekly.sort_index().loc['2019-01':]

# na to 1, 1 to na conversion
def flip(df):
    res = df.copy()
    res['signal'] = np.where(res['signal'] == 1, np.nan, 1)
    return res

def sum(df_result):
    res = df_result['summary']
    dummy = cum(df_result)
    res['maxret'] = dummy['cumret'].max()
    res['mdd'] = dummy['drawdown'].min()
    return res

def cum(df_result):
    res1 = df_result['daily_ret'].groupby(level = 1).sum().cumsum()
    res2 = res1 - res1.cummax()
    res = pd.concat([res1, res2], axis = 1)
    res.columns = ['cumret', 'drawdown']
    # res1.plot()
    # res2.plot(secondary_y = True)
    return res

def anycum(any_result):
    res1 = any_result
    res2 = any_result - any_result.cummax()
    res = pd.concat([res1, res2], axis = 1)
    res.columns = ['cumret', 'drawdown']
    # res1.plot()
    # res2.plot(secondary_y = True)
    return res

def plot(df_result):
    cum(res).plot()

def table(df_result):
    res = df_result['all_trades'].sort_values('final_ret', ascending = False)
    return res

def all(df_result):
    return sum(df_result), table(df_result), print(cum(df_result))

def vol_based_sizing(vkospi, multiplier = 1, vol_percentile = [0.5, 0.75]):
    vol_bins = [0] + vol_percentile + [1]
    trade_volume = [1 + i * multiplier for i in range(len(vol_bins) - 1)]
    vol_rank = vkospi['close'].rank(pct= True)
    res = pd.cut(vol_rank, bins = vol_bins, labels = trade_volume)
    res = res.astype('int64')
    return res

def stopbyme(df_result, profit_take, stop_loss):

    # number of trades whose profits are realized past profit taking threshold
    df_profit = table(df_result).loc[table(df_result)['final_ret'] > profit_take]
    number_of_profit = len(table(df_result).loc[table(df_result)['final_ret'] > profit_take])
    
    df_loss = table(df_result).loc[table(df_result)['final_ret'] < stop_loss]
    number_of_loss = len(table(df_result).loc[table(df_result)['final_ret'] < stop_loss])

    print("nmumber_of_profit : ", number_of_profit)
    print("number_of_loss : " , number_of_loss)
    
    return df_profit, df_loss

def scale(df_result, df_sizing):
    a = (df_result['all_trades']['trade_ret'] * df_sizing).dropna()
    aa = pd.concat(a.tolist(), axis = 1, ignore_index = True)
    aaa = aa.sum(axis = 1)
    res = aaa.cumsum()
    res = pd.DataFrame(res, columns = ['cumret'])
    res['drawdown'] = res - res.cummax()
    print(f"totalret : {res['cumret'].iloc[-1]}, mdd : {res['drawdown'].min()}, maxret : {res['cumret'].max()}")
    res.to_csv('./scaled_ret.csv')
    return res

# entry_date : 온갖 방법으로 entry date 도출

from get_entry_date import get_date_intersect, get_date_union, weekday_entry, change_recent, notrade, stoch_signal, rsi_signal, bband_signal, psar_signal, supertrend_signal

psar_turnup = k200.psar.rebound(pos = 'l')
psar_turndown = k200.psar.rebound(pos = 's')

psar_trendup = k200.psar.trend(pos = 'l')
psar_trenddown = k200.psar.trend(pos = 's')

supertrend_turnup = k200.supertrend.rebound(pos = 'l', length = 7, atr_multiplier = 3)
supertrend_turndown = k200.supertrend.rebound(pos = 's', length = 7, atr_multiplier = 3)

supertrend_trendup = k200.supertrend.trend(pos = 'l', length = 7, atr_multiplier = 3)
supertrend_trenddown = k200.supertrend.trend(pos = 's', length = 7, atr_multiplier = 3)

bbands_turnup1 = k200.bbands.through_bbands(pos = 'l', length = 20, std = 2)
bbands_turndown1 = k200.bbands.through_bbands(pos = 's', length = 20, std = 2)

bbands_turnup2 = k200.bbands.through_bbands(pos = 'l', length = 60, std = 2)
bbands_turndown2 = k200.bbands.through_bbands(pos = 's', length = 60, std = 2)

stoch_turnup1 = k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turndown1 = k200.stoch.rebound1(pos = 's', k = 10, d = 5, smooth_d = 5)

stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')


no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

lowvol_only1 = flip(notrade.vkospi_below_n(0.2))
lowvol_only2 = flip(notrade.vkospi_below_n(0.5))
highvol_only = flip(notrade.vkospi_above_n(0.2))

no_vixinvert = notrade.vix_curve_invert()

vol2 = notrade.vkospi_above_n(0.2)

vol3 = notrade.vkospi_below_n(0.2)
vol4 = notrade.vkospi_above_n(0.4)

vol5 = notrade.vkospi_below_n(0.4)
vol6 = notrade.vkospi_above_n(0.6)

vol7 = notrade.vkospi_below_n(0.6)
vol8 = notrade.vkospi_above_n(0.8)

vol9 = notrade.vkospi_below_n(0.8)

# 각종 진입조건 : 전략결과.xlsx 에 있음

#%% 양매매 test

no_lowvol = notrade.vkospi_below_n(0.2)
no_vixinvert = notrade.vix_curve_invert()

entry = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), no_vixinvert)
# strat = {'C': [('delta', 0.15, -1)], 'P': [('delta', -0.15, -1)]}
strat = {'C': [('delta', 0.15, -2), ('delta', 0.1, 2)], 'P': [('delta', -0.15, -2), ('delta', -0.1, 2)]}
# strat = {'C': [('number', 7.5, -1)], 'P': [('number', -7.5, -1)]}

exit = []
stop = 1
profit_take = 0.5
stop_loss = -0.5
dte_range = [2, 9]

res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = entry,
                                              trade_spec = strat,
                                              dte_range = dte_range,
                                              exit_dates = exit,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

vol = vol_based_sizing(vkospi, 1, [0.5, 0.8])
scaled_res = scale(res, vol)

stopbyme(res, profit_take, stop_loss)

print(sum(res))
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")


#%% 상승_test

entry = get_date_intersect(df_weekly, change_recent(k200, -0.03, 'close'))
# entry = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]))

strat = {'C': [('delta', 0.2, 1)]}
# strat = {'C': [('delta', 0.3, 1)]}
# exit = get_date_union(df_weekly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5))
exit = []
stop = 0
profit_take = 0.25
stop_loss = -0.5
dte_range = [2, 9]
 
res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = entry,
                                              trade_spec = strat,
                                              dte_range = dte_range,
                                              exit_dates = exit,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

# 변동성 scaling
vol = vol_based_sizing(vkospi, 1, [0.5, 0.8])
scaled_res = scale(res, vol)

stopbyme(res, profit_take, stop_loss)

print(sum(res))
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")


#%% 하락 test

entry = get_date_intersect(df_monthly, change_recent(k200, 0.05) * -1)
strat = {'P': [('delta', -0.3, -1), ('delta', -0.15, 2)]}
# exit =  get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5))
exit = []
stop = 1
profit_take = 0.5
stop_loss = -1
dte_range = [7, 35]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = entry,
                                              trade_spec = strat,
                                              dte_range = dte_range,
                                              exit_dates = exit,
                                              stop_dte = stop,
                                              is_complex_strat = True,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

# 변동성 scaling
vol = vol_based_sizing(vkospi, 1, [0.5, 0.8])
scaled_res = scale(res, vol)

stopbyme(res, profit_take, stop_loss)

print(sum(res))
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")

#%% BACKTEST_월물풋매도

#1. 진입조건

from itertools import product
import time

#1. 요일별 벡테스트

entry_condition = [
    dict(entry1 = get_date_intersect(df_monthly, change_recent(k200, -0.03, 'close'))),
    dict(entry2 = get_date_intersect(df_monthly, change_recent(k200, -0.04, 'close'))),
    dict(entry3 = get_date_intersect(df_monthly, change_recent(k200, -0.05, 'close'))),
    dict(entry4 = get_date_intersect(df_monthly, change_recent(k200, -0.08, 'close')))
]

# entry_condition = [
#     dict(entry1 = get_date_intersect(df_monthly, change_recent(k200, 0.03, 'close') * -1)),
#     dict(entry2 = get_date_intersect(df_monthly, change_recent(k200, 0.04, 'close') * -1)),
#     dict(entry3 = get_date_intersect(df_monthly, change_recent(k200, 0.05, 'close') * -1)),
#     dict(entry4 = get_date_intersect(df_monthly, change_recent(k200, 0.06, 'close') * -1)),
#     dict(entry5 = get_date_intersect(df_monthly, change_recent(k200, 0.08, 'close') * -1))
# ]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
strat= [
    # {'C' : [('delta', 0.4, 1)]},
    # {'C' : [('delta', 0.2, 1)]},
    # {'C' : [('delta', 0.3, 1), ('delta', 0.15, -1)]},
    # {'C' : [('delta', 0.2, 1), ('delta', 0.1, -1)]}
    # {'P' : [('delta', -0.4, -1)]},
    # {'P' : [('delta', -0.2, -1)]},
    # {'P' : [('delta', -0.3, -1), ('delta', -0.15, 1)]},
    # {'P' : [('delta', -0.2, -1), ('delta', -0.1, 1)]}
    # {'C' : [('delta', 0.4, 1), ('delta', 0.2, -2)]},
    # {'C' : [('delta', 0.3, 1), ('delta', 0.15, -2)]},
    # {'C' : [('delta', 0.2, 1), ('delta', 0.1, -2)]}
    # {'C' : [('delta', 0.4, -1), ('delta', 0.2, 2)]},
    # {'C' : [('delta', 0.3, -1), ('delta', 0.15, 2)]},
    # {'C' : [('delta', 0.2, -1), ('delta', 0.1, 2)]}
]

# strat= [
#     # {'P' : [('delta', -0.4, 1)]},
#     # {'P' : [('delta', -0.2, 1)]},
#     # {'P' : [('delta', -0.3, 1), ('delta', -0.15, -1)]},
#     # {'P' : [('delta', -0.2, 1), ('delta', -0.1, -1)]}
#     # {'C' : [('delta', 0.4, -1)]},
#     # {'C' : [('delta', 0.2, -1)]},
#     # {'C' : [('delta', 0.3, -1), ('delta', 0.15, 1)]},
#     # {'C' : [('delta', 0.2, -1), ('delta', 0.1, 1)]}
#     # {'P' : [('delta', -0.4, 1), ('delta', -0.2, -2)]},
#     # {'P' : [('delta', -0.3, 1), ('delta', -0.15, -2)]},
#     # {'P' : [('delta', -0.2, 1), ('delta', -0.1, -2)]}
#     # {'P' : [('delta', -0.4, -1), ('delta', -0.2, 2)]},
#     # {'P' : [('delta', -0.3, -1), ('delta', -0.15, 2)]},
#     # {'P' : [('delta', -0.2, -1), ('delta', -0.1, 2)]}
# ]

#3. 어떤 만기 종목
dte_range = [
            [7, 35]
             ]

#4. 청산 조건
exit_condition = [
    dict(exit1 = [])
]

#5. 익절
profit_target = [0.25, 0.5, 1, 2, 4, 999] # 매수, 절대값
profit_target = [0.25, 0.5, 0.8] # 매도

# 6. 손절
stop_loss = [-0.25, -0.5, -0.8] # 매수
stop_loss = [-0.25, -0.5, -1, -2] # 매도
stop_loss = [-0.25, -0.5, -1, -2, -3] # 절대값

comb = list(product(entry_condition, strat, dte_range, exit_condition, profit_target, stop_loss))

for i in range(0, len(comb), 100):

    df_res = dict()
    chunk = comb[i:i+100]

    for entry, trade, dte, exit, profit_target, stop_loss in chunk:
        start = time.time()
        entry_name = list(entry.keys())[0]
        entry_value = list(entry.values())[0]
        exit_name = list(exit.keys())[0]
        exit_value = list(exit.values())[0]
        res = backtest.get_vertical_trade_result(df_monthly,
                                entry_dates = entry_value,
                                trade_spec = trade,
                                dte_range = dte,
                                exit_dates = exit_value,
                                stop_dte = 1,
                                is_complex_strat = True,
                                profit_take = profit_target,
                                stop_loss = stop_loss)
        result = dict(
        n = sum(res)['n'],
        win = sum(res)['win'],
        totalret = sum(res)['total_ret'],
        maxret = cum(res)['cumret'].max(),
        mdd = cum(res)['drawdown'].min()
        )

        df_res[f"{entry_name}_{trade}_{dte}_{exit_name}_{profit_target}_{stop_loss}"] = result
        end = time.time()
        print(start - end)
        
    csv_res = pd.DataFrame(df_res).T
    csv_res.to_csv(f"./res_dump/{i}_{i} + 100.csv")
    del df_res
    del chunk
#%% BACKTEST_월물풋매수

#1. 진입조건

from itertools import product
import time

entry_condition = [

    dict(entry18 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown, lowvol_only1)),
    dict(entry19 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown, lowvol_only1)),
    dict(entry20 = get_date_intersect(df_monthly, psar_turndown, lowvol_only1)),
    dict(entry21 = get_date_intersect(df_monthly, supertrend_turndown, lowvol_only1)),
    dict(entry22 = get_date_intersect(df_monthly, bbands_turndown1, lowvol_only1)),
    dict(entry23 = get_date_intersect(df_monthly, bbands_turndown2, lowvol_only1)),
    dict(entry24 = get_date_intersect(df_monthly, stoch_turndown1, lowvol_only1)),
    dict(entry25 = get_date_intersect(df_monthly, stoch_turndown2, lowvol_only1)),
    dict(entry26 = get_date_intersect(df_monthly, rsi_turndown, lowvol_only1)),

    dict(entry27 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown, lowvol_only2)),
    dict(entry28 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown, lowvol_only2)),
    dict(entry29 = get_date_intersect(df_monthly, psar_turndown, lowvol_only2)),
    dict(entry30 = get_date_intersect(df_monthly, supertrend_turndown, lowvol_only2)),
    dict(entry31 = get_date_intersect(df_monthly, bbands_turndown1, lowvol_only2)),
    dict(entry32 = get_date_intersect(df_monthly, bbands_turndown2, lowvol_only2)),
    dict(entry33 = get_date_intersect(df_monthly, stoch_turndown1, lowvol_only2)),
    dict(entry34 = get_date_intersect(df_monthly, stoch_turndown2, lowvol_only2)),
    dict(entry35 = get_date_intersect(df_monthly, rsi_turndown, lowvol_only2))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
strat= [
    {'P' : [('delta', -0.2, 1)]},
    {'P' : [('delta', -0.4, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35],
             ]

#4. 청산 조건
exit_condition = [
    dict(exit1 = []),
    dict(exit2 = get_date_intersect(df_monthly, psar_turnup)),
    dict(exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))),
    dict(exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [1, 2, 4]
#6. 손절
stop_loss = [-0.25, -0.5]

comb = list(product(entry_condition, strat, dte_range, exit_condition, profit_target, stop_loss))

for i in range(0, len(comb), 100):

    df_res = dict()
    chunk = comb[i:i+100]

    for entry, trade, dte, exit, profit_target, stop_loss in chunk:
        start = time.time()
        entry_name = list(entry.keys())[0]
        entry_value = list(entry.values())[0]
        exit_name = list(exit.keys())[0]
        exit_value = list(exit.values())[0]
        res = backtest.get_vertical_trade_result(df_monthly,
                                entry_dates = entry_value,
                                trade_spec = trade,
                                dte_range = dte,
                                exit_dates = exit_value,
                                stop_dte = 0,
                                is_complex_strat = False,
                                profit_take = profit_target,
                                stop_loss = stop_loss)
        result = dict(
        n = sum(res)['n'],
        win = sum(res)['win'],
        totalret = sum(res)['total_ret'],
        maxret = cum(res)['cumret'].max(),
        mdd = cum(res)['drawdown'].min()
        )

        df_res[f"{entry_name}_{trade}_{dte}_{exit_name}_{profit_target}_{stop_loss}"] = result
        end = time.time()
        print(start - end)
        
    csv_res = pd.DataFrame(df_res).T
    csv_res.to_csv(f"./buyput/{i}_{i} + 100.csv")
    del df_res
    del chunk
