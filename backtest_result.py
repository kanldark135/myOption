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
    print(f"mdd : {res['drawdown'].min()}, maxret : {res['cumret'].max()}")
    res.to_csv('./scaled_ret.csv')
    return res

# entry_date : 온갖 방법으로 entry date 도출

from get_entry_date import get_date_intersect, get_date_union, weekday_entry, notrade, stoch_signal, rsi_signal, bband_signal, psar_signal, supertrend_signal


#%% weekly_strangle_test


no_vixinvert = notrade.vix_curve_invert()
lowvol_only = notrade.vkospi_above_n(0.5)
no_highvol = notrade.vkospi_above_n(0.8)

# 조정순서
#1. 진입시점 (put_entry) : 전략별로 다르게 (entry)
#2. 델타 (trade[0, 1, 2...] :어짜피 그게 그거라는 생각, 40으로 고정)
#3. 손절컷 (stop_loss : -1/-2)6y 
#4. 조기엑싯 (exit : noexit / ...)
# 변동성 사이징은 눈으로 보면서 판단

strangle = {'C': [('number', 2.5, 1)], 'P': [('number', -2.5, 1)]}

strangle_entry = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), lowvol_only)


exit1 = []
stop = 0
profit_take = 1
stop_loss = -0.1
dte_range = [2, 9]

res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = strangle_entry,
                                              trade_spec = strangle,
                                              dte_range = dte_range,
                                              exit_dates = exit1,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% strangle_test

no_vixinvert = notrade.vix_curve_invert()
lowvol_only = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

# 조정순서
#1. 진입시점 (put_entry) : 전략별로 다르게 (entry)
#2. 델타 (trade[0, 1, 2...] :어짜피 그게 그거라는 생각, 40으로 고정)
#3. 손절컷 (stop_loss : -1/-2)6y 
#4. 조기엑싯 (exit : noexit / ...)
# 변동성 사이징은 눈으로 보면서 판단

strangle_entry4 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), lowvol_only)

strangle = {'C': [('delta', 0.06, -1)], 'P': [('delta', -0.07, -1)]}

exit1 = []

stop = 1
profit_take = 0.5
stop_loss = -1
dte_range = [42, 70]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = strangle_entry4,
                                              trade_spec = strangle,
                                              dte_range = dte_range,
                                              exit_dates = exit1,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% weekly_put_test

psar_turnup = k200.psar.rebound(pos = 'l')
psar_turndown = k200.psar.rebound(pos = 's')

psar_trendup = k200.psar.trend(pos = 'l')
psar_trenddown = k200.psar.trend(pos = 's')

supertrend_turnup = k200.supertrend.rebound(pos = 'l', length = 7, atr_multiplier = 3)
supertrend_turndown = k200.supertrend.rebound(pos = 's', length = 7, atr_multiplier =
                                               3)
supertrend_trendup = k200.supertrend.trend(pos = 'l', length = 7, atr_multiplier = 3)
supertrend_trenddown = k200.supertrend.trend(pos = 's', length = 7, atr_multiplier = 3)

bbands_turnup1 = k200.bbands.through_bbands(pos = 'l', length = 20, std = 2)
bbands_turndown1 = k200.bbands.through_bbands(pos = 's', length = 20, std = 2)
bbands_turnup2 = k200.bbands.through_bbands(pos = 'l', length = 60, std = 2)
bbands_turndown2 = k200.bbands.through_bbands(pos = 's', length = 60, std = 2)

stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turndown1 = k200.stoch.rebound1(pos = 's', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')

no_vixinvert = notrade.vix_curve_invert()
lowvol_only = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

#1. 진입조건
put_entry = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), psar_trenddown)

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

put_strat = {'P': [('delta', -0.50, 1)]}

dte_range = [2, 9]

put_exit = []

put_stop = 0
profit_take = 999
stop_loss = -1

res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = put_entry,
                                              trade_spec = put_strat,
                                              dte_range = dte_range,
                                              exit_dates = put_exit,
                                              stop_dte = put_stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% put_test

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
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)
stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')

no_vixinvert = notrade.vix_curve_invert()
lowvol_only = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

put_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)
put_entry4 = get_date_intersect(df_monthly, supertrend_turndown)
put_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)

buy_putbs = {"P" : [('delta', -0.5, -1), ('delta', -0.26, 2)]}

put_exit1 = []
put_exit2 = get_date_intersect(df_monthly, psar_turnup)
put_exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))
put_exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5, smooth_d = 5))

put_stop = 1
profit_take = 0.5
stop_loss = -999
dte_range = [42, 71]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = put_entry2,
                                              trade_spec = buy_putbs,
                                              dte_range = dte_range,
                                              exit_dates = put_exit4,
                                              stop_dte = put_stop,
                                              is_complex_strat = True,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% weekly_call_test

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

stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turndown1 = k200.stoch.rebound1(pos = 's', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')

call_entry =  get_date_intersect(df_weekly, psar_turnup)
call_strat = {'C': [('delta', 0.25, 1)]}

dte_range = [2, 9]

call_exit1 = []

call_stop = 0
profit_take = 2
stop_loss = -0.2

res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = call_entry,
                                              trade_spec = call_strat,
                                              dte_range = dte_range,
                                              exit_dates = call_exit1,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% call_test

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
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)
stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')

no_vixinvert = notrade.vix_curve_invert()
lowvol_only = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

# call_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)
# call_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)
# call_entry3 = get_date_intersect(df_monthly, psar_turnup)
# call_entry4 = get_date_intersect(df_monthly, supertrend_turnup)
# call_entry5 = get_date_intersect(df_monthly, bbands_turnup1)
# call_entry6 = get_date_intersect(df_monthly, bbands_turnup2)
# call_entry7 = get_date_intersect(df_monthly, stoch_turnup1)
# call_entry8 = get_date_intersect(df_monthly, stoch_turnup2)
# call_entry9 = get_date_intersect(df_monthly, rsi_turnup)

call_entry = get_date_intersect(df_monthly, supertrend_turnup)

call_strat = {'C': [('delta', 0.4, 1)]}

dte_range = [7, 36]

call_exit = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5, smooth_d = 5))

# call_exit2 = get_date_intersect(df_monthly, psar_turndown)
# call_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))
# call_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5, smooth_d = 5))

call_stop = 1
profit_take = 999
stop_loss = -0.25

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry,
                                              trade_spec = call_strat,
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)
#%% backtest

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
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)
stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')


#1. 진입조건
entry_condition = [
    dict(call_entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), psar_trendup)),
    dict(call_entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), supertrend_trendup)),
    dict(call_entry3 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trendup)),
    dict(call_entry4 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trendup)),
    dict(call_entry5 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trendup)),
    dict(call_entry6 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trendup)),
    dict(call_entry7 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), psar_trendup)),
    dict(call_entry8 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), supertrend_trendup)),
    dict(call_entry9 = get_date_intersect(df_weekly, psar_turnup)),
    dict(call_entry10 = get_date_intersect(df_weekly, supertrend_turnup)),
    dict(call_entry11 = get_date_intersect(df_weekly, bbands_turnup1)),
    dict(call_entry12 = get_date_intersect(df_weekly, bbands_turnup2)),
    dict(call_entry13 = get_date_intersect(df_weekly, stoch_turnup1)),
    dict(call_entry14 = get_date_intersect(df_weekly, stoch_turnup2))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
buy_call= [
    {'C' : [('delta', 0.5, 1)]},
    {'C' : [('delta', 0.25, 1)]},
    {'C' : [('delta', 0.10, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [2, 9]
             ]

#4. 청산 조건
exit_condition = [
    dict(call_exit1 = [])
    ]

#5. 익절 
profit_target = [0.2, 0.5, 1, 2, 999]
#6. 손절
stop_loss = [-0.2, -0.5, -1]

comb = list(product(entry_condition, buy_call, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_call/{i}_{i} + 100.csv")
    del df_res
    del chunk

#%% 풋매수 위클리
    
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

stoch_turnup1= k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
stoch_turndown1 = k200.stoch.rebound1(pos = 's', k = 10, d = 5, smooth_d = 5)
stoch_turnup2 = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)
stoch_turndown2 = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)

rsi_turnup = k200.rsi.rebound(pos = 'l')
rsi_turndown = k200.rsi.rebound(pos = 's')


#1. 진입조건
entry_condition = [
    dict(put_entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), psar_trenddown)),
    dict(put_entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), supertrend_trenddown)),
    dict(put_entry3 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trenddown)),
    dict(put_entry4 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trenddown)),
    dict(put_entry5 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trenddown)),
    dict(put_entry6 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trenddown)),
    dict(put_entry7 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), psar_trenddown)),
    dict(put_entry8 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [1]), supertrend_trenddown)),
    dict(put_entry9 = get_date_intersect(df_weekly, psar_turndown)),
    dict(put_entry10 = get_date_intersect(df_weekly, supertrend_turndown)),
    dict(put_entry11 = get_date_intersect(df_weekly, bbands_turndown1)),
    dict(put_entry12 = get_date_intersect(df_weekly, bbands_turndown2)),
    dict(put_entry13 = get_date_intersect(df_weekly, stoch_turndown1)),
    dict(put_entry14 = get_date_intersect(df_weekly, stoch_turndown2)),
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)


buy_put =[
    {'P': [('delta', -0.5, 1)]},
    {'P': [('delta', -0.25, 1)]},
    {'P': [('delta', -0.10, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [2, 9]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = [])
]

#5. 익절 
profit_target = [0.2, 0.5, 1, 2, 999]
#6. 손절
stop_loss = [-0.2, -0.5, -1]

comb = list(product(entry_condition, buy_put, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_put/{i}_{i} + 100.csv")
    del df_res
    del chunk
