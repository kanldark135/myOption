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

from get_entry_date import get_date_intersect, get_date_union, weekday_entry, notrade, stoch_signal, rsi_signal, bband_signal, psar_signal, supertrend_signal

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
no_highvol = notrade.vkospi_above_n(0.2)

lowvol_only1 = flip(notrade.vkospi_below_n(0.2))
lowvol_only2 = flip(notrade.vkospi_below_n(0.5))
highvol_only = flip(notrade.vkospi_above_n(0.2))

no_vixinvert = notrade.vix_curve_invert()

# 상승진입조건

entry_condition = [
    dict(entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)),
    dict(entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)),
    dict(entry3 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [1]), psar_trendup)),
    dict(entry4 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [1]), supertrend_trendup)),
    dict(entry5 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [2]), psar_trendup)),
    dict(entry6 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [2]), supertrend_trendup)),
    dict(entry7 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [3]), psar_trendup)),
    dict(entry8 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [3]), supertrend_trendup)),
    dict(entry9 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [4]), psar_trendup)),
    dict(entry10 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [4]), supertrend_trendup)),
    dict(entry11 = get_date_intersect(df_monthly, psar_turnup)),
    dict(entry12 = get_date_intersect(df_monthly, supertrend_turnup)),
    dict(entry13 = get_date_intersect(df_monthly, bbands_turnup1)),
    dict(entry14 = get_date_intersect(df_monthly, bbands_turnup2)),
    dict(entry15 = get_date_intersect(df_monthly, stoch_turnup1)),
    dict(entry16 = get_date_intersect(df_monthly, stoch_turnup2)),
    dict(entry17 = get_date_intersect(df_monthly, rsi_turnup))
]

# exit_condition = [
#     dict(exit1 = []),
#     dict(exit2 = get_date_intersect(df_monthly, psar_turndown)),
#     dict(exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))),
#     dict(exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5)))
# ]


# 하락진입조건
entry_condition = [
    dict(entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)),
    dict(entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)),
    dict(entry3 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [1]), psar_trenddown)),
    dict(entry4 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [1]), supertrend_trenddown)),
    dict(entry5 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [2]), psar_trenddown)),
    dict(entry6 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [2]), supertrend_trenddown)),
    dict(entry7 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [3]), psar_trenddown)),
    dict(entry8 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [3]), supertrend_trenddown)),
    dict(entry9 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [4]), psar_trenddown)),
    dict(entry10 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [4]), supertrend_trenddown)),
    dict(entry11 = get_date_intersect(df_monthly, psar_turndown)),
    dict(entry12 = get_date_intersect(df_monthly, supertrend_turndown)),
    dict(entry13 = get_date_intersect(df_monthly, bbands_turndown1)),
    dict(entry14 = get_date_intersect(df_monthly, bbands_turndown2)),
    dict(entry15 = get_date_intersect(df_monthly, stoch_turndown1)),
    dict(entry16 = get_date_intersect(df_monthly, stoch_turndown2)),
    dict(entry17 = get_date_intersect(df_monthly, rsi_turndown)),
]

exit_condition = [
    dict(exit1 = []),
    dict(exit2 = get_date_intersect(df_monthly, psar_turnup)),
    dict(exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))),
    dict(exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5)))
]

# # 양매도
# entry_condition = [
#     dict(entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]))),
#     dict(entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_vixinvert)),
#     dict(entry3 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_lowvol)),
#     dict(entry4 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_highvol)),
#     dict(entry5 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_lowvol, no_vixinvert))
# ]

# # 양매수
# entry_condition = [
#     dict(entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]))),
#     dict(entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), lowvol_only1)),
#     dict(entry3 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), lowvol_only2)),
# ]
 

#%% strangle_test

strangle_entry = get_date_intersect(df_monthly, weekday_entry(df_monthly, [4]), lowvol_only1)
strangle = {'C': [('delta', 0.5, 1)], 'P': [('delta', -0.5, 1)]}
exit = []
stop = 0
profit_take = 2
stop_loss = -0.5
dte_range = [7, 35]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = strangle_entry,
                                              trade_spec = strangle,
                                              dte_range = dte_range,
                                              exit_dates = exit,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

vol = vol_based_sizing(vkospi, 1, [0.5, 0.8])
scaled_res = scale(res, vol)

sum(res)
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")



#%% test

entry =  get_date_intersect(df_monthly, bbands_turndown2)
strat = {'P': [('delta', -0.3, -1), ('delta', -0.15, 2)]}
exit = get_date_union(df_monthly, psar_turnup)
# exit = []
stop = 1
profit_take = 2
stop_loss = -0.5
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

sum(res)
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")



#%% call_test

entry = get_date_intersect(df_monthly, bbands_turnup2)
strat = {'C': [('delta', 0.3, -1), ('delta', 0.15, 2)]}
# exit = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5))
exit = []
stop = 1
profit_take = 6
stop_loss = -3
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

sum(res)
plot(res)
cum(res).drop(columns = ['drawdown']).to_csv("./ret.csv")
scaled_res.drop(columns = ['drawdown']).to_csv("./scaled_ret.csv")

#%% 상승 backtest

#1. 진입조건

from itertools import product
import time

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

stoch_turndown1 = k200.stoch.rebound1(pos = 's', k = 10, d = 5, smooth_d = 5)
stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')


#1. 진입조건
entry_condition = [
    dict(entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trendup)),
    dict(entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trendup)),
    dict(entry3 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), psar_trendup)),
    dict(entry4 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), supertrend_trendup)),
    dict(entry5 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [2]), psar_trendup)),
    dict(entry6 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [2]), supertrend_trendup)),
    dict(entry7 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), psar_trendup)),
    dict(entry8 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), supertrend_trendup)),
    dict(entry9 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trendup)),
    dict(entry10 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trendup)),
    dict(entry11 = get_date_intersect(df_weekly, psar_turnup)),
    dict(entry12 = get_date_intersect(df_weekly, supertrend_turnup)),
    dict(entry13 = get_date_intersect(df_weekly, bbands_turnup1)),
    dict(entry14 = get_date_intersect(df_weekly, bbands_turnup2)),
    dict(entry15 = get_date_intersect(df_weekly, stoch_turnup1)),
    dict(entry16 = get_date_intersect(df_weekly, stoch_turnup2)),
    dict(entry17 = get_date_intersect(df_weekly, rsi_turnup))
]


#1. 진입조건
# entry_condition = [
#     dict(entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), psar_trenddown)),
#     dict(entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), supertrend_trenddown)),
#     dict(entry3 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trenddown)),
#     dict(entry4 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trenddown)),
#     dict(entry5 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trenddown)),
#     dict(entry6 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trenddown)),
#     dict(entry7 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), psar_trenddown)),
#     dict(entry8 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), supertrend_trenddown)),
#     dict(entry9 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [2]), psar_trenddown)),
#     dict(entry10 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [2]), supertrend_trenddown)),
#     dict(entry11 = get_date_intersect(df_weekly, psar_turndown)),
#     dict(entry12 = get_date_intersect(df_weekly, supertrend_turndown)),
#     dict(entry13 = get_date_intersect(df_weekly, bbands_turndown1)),
#     dict(entry14 = get_date_intersect(df_weekly, bbands_turndown2)),
#     dict(entry15 = get_date_intersect(df_weekly, stoch_turndown1)),
#     dict(entry16 = get_date_intersect(df_weekly, stoch_turndown2)),
#     dict(entry17 = get_date_intersect(df_weekly, rsi_turndown)),
# ]


#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
strat= [
    {'P' : [('delta', -0.5, 1)]},
    {'P' : [('delta', -0.25, 1)]},
    {'P' : [('delta', -0.10, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [2, 9]
             ]

#4. 청산 조건
exit_condition = [
    dict(exit1 = []),
    dict(exit2 = get_date_intersect(df_monthly, psar_turndown)),
    dict(exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))),
    dict(exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5)))
]


#5. 익절 
profit_target = [0.2, 0.5, 1, 2, 999]
#6. 손절
stop_loss = [-0.2, -0.5, -1]

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
        res = backtest.get_vertical_trade_result(df_weekly,
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
    csv_res.to_csv(f"./res_dump/{i}_{i} + 100.csv")
    del df_res
    del chunk

#%% 
from itertools import product
import time

lowvol_1 = notrade.vkospi_above_n(0.2)
lowvol_2 = notrade.vkospi_above_n(0.5)

#1. 진입조건
entry_condition = [
    dict(strangle_entry4 = get_date_intersect(df_monthly, lowvol_1, weekday_entry(df_monthly, [1]))),
    dict(strangle_entry5 = get_date_intersect(df_monthly, lowvol_1, weekday_entry(df_monthly, [2]))),
    dict(strangle_entry6 = get_date_intersect(df_monthly, lowvol_1, weekday_entry(df_monthly, [3]))),
    dict(strangle_entry7 = get_date_intersect(df_monthly, lowvol_1, weekday_entry(df_monthly, [4])))
    ]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

buy_strangle =[
    {'C': [('delta', 0.5, 1)], "P" : [('delta', -0.5, 1)]},
    {'C': [('delta', 0.3, 1)], "P" : [('delta', -0.3, 1)]},
    {'C': [('delta', 0.15, 1)], "P" : [('delta', -0.15, 1)]},
    {'C': [('delta', 0.06, 1)], "P" : [('delta', -0.07, 1)]},
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35]
             ]

#4. 청산 조건
exit_condition = [
    dict(exit1 = [])
    ]

#5. 익절 
profit_target = [0.1, 0.25, 0.5, 1, 2, 4]
#6. 손절
stop_loss = [-0.1, -0.25, -0.5, -1]

comb = list(product(entry_condition, buy_strangle, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump/{i}_{i} + 100.csv")
    del df_res
    del chunk


