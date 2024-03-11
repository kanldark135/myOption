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


# long call
buy_call =[
    {'C': [('delta', 0.4, 1)]},
    {'C': [('delta', 0.3, 1)]},
    {'C': [('delta', 0.2, 1)]},
    {'C': [('delta', 0.1, 1)]}
]

# long call spreads
buy_call_debit = [
    {'C' : [('number', 0, 1), ('number', 2.5, -1)]},
    {'C' : [('number', 2.5, 1), ('number', 5, -1)]},
    {'C' : [('number', 5, 1), ('number', 7.5, -1)]},
    {'C' : [('number', 7.5, 1), ('number', 10, -1)]}
]

buy_call_backspread = [
    {'C' : [('delta', 0.5, -1), ('delta', 0.25, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 3)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 2)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 3)]}
]
sell_call_front = {'C' : [('delta', 0.4, -1)]}
buy_call_back = {'C' : [('delta', 0.2, 2)]} 


sell_call =[
    {'C': [('delta', 0.4, -1)]},
    {'C': [('delta', 0.3, -1)]},
    {'C': [('delta', 0.2, -1)]},
    {'C': [('delta', 0.1, -1)]}
]

sell_call_credit = [
    {'C' : [('number', 0, -1), ('number', 2.5, 1)]},
    {'C' : [('number', 2.5, -1), ('number', 5, 1)]},
    {'C' : [('number', 5, -1), ('number', 7.5, 1)]},
    {'C' : [('number', 7.5, -1), ('number', 10, 1)]}
]

sell_call_ratio = [
    {'C' : [('delta', 0.5, 1), ('delta', 0.25, -2)]},
    {'C' : [('delta', 0.4, 1), ('delta', 0.2, -2)]},
    {'C' : [('delta', 0.4, 1), ('delta', 0.2, -3)]},
    {'C' : [('delta', 0.3, 1), ('delta', 0.15, -2)]},
    {'C' : [('delta', 0.3, 1), ('delta', 0.15, -3)]}
]

sell_call_111 = [
    {'C' : [('delta', 0.5, 1), ('delta', 0.46, -1), ('delta', 0.15, -1)]},
    {'C' : [('delta', 0.4, 1), ('delta', 0.36, -1), ('delta', 0.125, -1)]},
    {'C' : [('delta', 0.3, 1), ('delta', 0.26, -1), ('delta', 0.10, -1)]},
    {'C' : [('delta', 0.25, 1), ('delta', 0.20, -1), ('delta', 0.08, -1)]},
    {'C' : [('delta', 0.2, 1), ('delta', 0.15, -1), ('delta', 0.07, -1)]}
]

#풋매수계열 진입 ---------------

buy_put = [{'P': [('delta', -0.4, 1)]},
            {'P': [('delta', -0.3, 1)]},
            {'P': [('delta', -0.2, 1)]},
            {'P': [('delta', -0.1, 1)]}
]

buy_put_debit = [
    {'P': [('number', 0, 1), ('number', -2.5, -1)]},
    {'P': [('number', -2.5, 1), ('number', -5, -1)]},
    {'P': [('number', -5, 1), ('number', -7.5, -1)]},
    {'P': [('number', -7.5, 1), ('number', -10, -1)]}
]

buy_put_backspread =[
    {'P': [('delta', -0.5, -1), ('delta', -0.26, 2)]},
    {'P': [('delta', -0.4, -1), ('delta', -0.21, 2)]},
    {'P': [('delta', -0.3, -1), ('delta', -0.16, 2)]},
    {'P': [('number', 0, -1), ('number', -5, 2)]},
    {'P': [('number', 0, -1), ('number', -7.5, 2)]}
]

sell_put = [{'P': [('delta', -0.4, -1)]},
            {'P': [('delta', -0.3, -1)]},
            {'P': [('delta', -0.2, -1)]},
            {'P': [('delta', -0.1, -1)]}
]

sell_put_credit = [
    {'P': [('number', 0, -1), ('number', -2.5, 1)]},
    {'P': [('number', -2.5, -1), ('number', -5, 1)]},
    {'P': [('number', -5, -1), ('number', -7.5, 1)]},
    {'P': [('number', -7.5, -1), ('number', -10, 1)]}
]

sell_put_111 = [
    {'P' : [('delta', -0.5, 1), ('delta', -0.46, -1), ('delta', -0.20, -1)]},
    {'P' : [('delta', -0.4, 1), ('delta', -0.36, -1), ('delta', -0.17, -1)]},
    {'P' : [('delta', -0.3, 1), ('delta', -0.26, -1), ('delta', -0.14, -1)]},
    {'P' : [('delta', -0.25, 1), ('delta',- 0.20, -1), ('delta',-0.10, -1)]},
    {'P' : [('delta', -0.2, 1), ('delta', -0.15, -1), ('delta', -0.07, -1)]}
]

buy_put_front = {'P': [('delta', -0.4, 1)]}
sell_put_back = {'P': [('delta', -0.2, -2)]} 

#%% weekly_strangle_test

no_vixinvert = notrade.vix_curve_invert()
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

# 조정순서
#1. 진입시점 (put_entry) : 전략별로 다르게 (entry)
#2. 델타 (trade[0, 1, 2...] :어짜피 그게 그거라는 생각, 40으로 고정)
#3. 손절컷 (stop_loss : -1/-2)6y 
#4. 조기엑싯 (exit : noexit / ...)
# 변동성 사이징은 눈으로 보면서 판단

strangle = {'C' : [('number', 7.5, -1)], "P" : [('number', -7.5, -1)]}

entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0, 4]), no_vixinvert)

exit1 = []

stop = 1
profit_take = 0.5
stop_loss = -0.5
dte_range = [1, 9]

res = backtest.get_vertical_trade_result(df_weekly,
                                              entry_dates = entry1,
                                              trade_spec = strangle,
                                              dte_range = dte_range,
                                              exit_dates = exit1,
                                              stop_dte = stop,
                                              is_complex_strat = False,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% strangle_test

no_vixinvert = notrade.vix_curve_invert()
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

# 조정순서
#1. 진입시점 (put_entry) : 전략별로 다르게 (entry)
#2. 델타 (trade[0, 1, 2...] :어짜피 그게 그거라는 생각, 40으로 고정)
#3. 손절컷 (stop_loss : -1/-2)6y 
#4. 조기엑싯 (exit : noexit / ...)
# 변동성 사이징은 눈으로 보면서 판단

strangle = {'C' : [('delta', 0.06, -1)], "P" : [('delta', -0.07, -1)]}

entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0, 4]), no_vixinvert, no_lowvol)

exit1 = []

stop = 0
profit_take = 0.1
stop_loss = -0.25
dte_range = [42, 70]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = entry1,
                                              trade_spec = strangle,
                                              dte_range = dte_range,
                                              exit_dates = exit1,
                                              stop_dte = stop,
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
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

put_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)
put_entry4 = get_date_intersect(df_monthly, supertrend_turndown)
put_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)

buy_putbs = {"P" : [('delta', -0.2, -1), ('delta', -0.1, 2)]}

put_exit1 = []
put_exit2 = get_date_intersect(df_monthly, psar_turndown)
put_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))
put_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5, smooth_d = 5))

put_stop = 1
profit_take = 999
stop_loss = -1
dte_range = [7, 36]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = put_entry4,
                                              trade_spec = buy_putbs,
                                              dte_range = dte_range,
                                              exit_dates = put_exit4,
                                              stop_dte = put_stop,
                                              is_complex_strat = True,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)

#%% callside_test

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
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

# 조정순서
#1. 진입기준조건  
#2. 델타(=행사가/무슨종목) 
#3. 만기선택
#4. 엑싯기준조건 
#5. 익손절 기준금액 
#6. 포지션 규모 (=사이징)

call_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)
call_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)
call_entry3 = get_date_intersect(df_monthly, psar_turnup)
call_entry4 = get_date_intersect(df_monthly, supertrend_turnup)
call_entry5 = get_date_intersect(df_monthly, bbands_turnup1)
call_entry6 = get_date_intersect(df_monthly, bbands_turnup2)
call_entry7 = get_date_intersect(df_monthly, stoch_turnup1)
call_entry8 = get_date_intersect(df_monthly, stoch_turnup2)
call_entry9 = get_date_intersect(df_monthly, rsi_turnup)

call_exit1 = []
call_exit2 = get_date_intersect(df_monthly, psar_turndown)
call_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))
call_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5, smooth_d = 5))

call_strat = {'C': [('delta', 0.4, -1), ('delta', 0.2, 3)]}

call_stop = 1
profit_take = 999
stop_loss = -3
dte_range = [42, 71]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry4,
                                              trade_spec = call_strat,
                                              dte_range = dte_range,
                                              exit_dates = call_exit4,
                                              stop_dte = call_stop,
                                              is_complex_strat = True,
                                              profit_take = profit_take,
                                              stop_loss = stop_loss)


#%% 스트랭글
   
from itertools import product
import time

no_vixinvert = notrade.vix_curve_invert()
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

#1. 진입조건
entry_condition = [
    dict(strangle_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]))),
    dict(strangle_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0, 4]))),
    dict(strangle_entry3 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_vixinvert)),
    dict(strangle_entry4 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_lowvol)),
    dict(strangle_entry5 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), no_highvol))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

sell_strangle =[
    {'C': [('delta', 0.3, -1)], "P" : [('delta', -0.3, -1)]},
    {'C': [('delta', 0.15, -1)], "P" : [('delta', -0.15, -1)]},
    {'C': [('delta', 0.06, -1)], "P" : [('delta', -0.07, -1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35], 
            [42, 70]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = [])
    ]

#5. 익절 
profit_target = [0.1, 0.25, 0.5, 0.8]
#6. 손절
stop_loss = [-0.25, -0.5, -1, -2]

comb = list(product(entry_condition, sell_strangle, dte_range, exit_condition, profit_target, stop_loss))

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


#%% 스트랭글 위클리
   
from itertools import product
import time

no_vixinvert = notrade.vix_curve_invert()
no_lowvol = notrade.vkospi_below_n(0.2)
no_highvol = notrade.vkospi_above_n(0.8)

#1. 진입조건
entry_condition = [
    # dict(strangle_entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]))),
    dict(strangle_entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0, 4])))
    # dict(strangle_entry3 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), no_vixinvert)),
    # dict(strangle_entry4 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), no_lowvol)),
    # dict(strangle_entry5 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), no_highvol))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
sell_weekly_strangle =[
    {'C': [('number', 10, -1)], "P" : [('number', -10, -1)]},
    {'C': [('number', 7.5, -1)], "P" : [('number', -7.5, -1)]},
    {'C': [('delta', 0.10, -1)], "P" : [('delta', -0.10, -1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [1, 9],
             ]

#4. 청산 조건
exit_condition = [
    dict(exit1 = [])
    ]

#5. 익절 
profit_target = [0.1, 0.25, 0.5, 0.8]
#6. 손절
stop_loss = [-0.25, -0.5, -1, -2]

comb = list(product(entry_condition, sell_weekly_strangle, dte_range, exit_condition, profit_target, stop_loss))

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
                                stop_dte = 1,
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


#%% 콜매수
   
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
    dict(call_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)),
    dict(call_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)),
    dict(call_entry3 = get_date_intersect(df_monthly, psar_turnup)),
    dict(call_entry4 = get_date_intersect(df_monthly, supertrend_turnup)),
    dict(call_entry5 = get_date_intersect(df_monthly, bbands_turnup1)),
    dict(call_entry6 = get_date_intersect(df_monthly, bbands_turnup2)),
    dict(call_entry7 = get_date_intersect(df_monthly, stoch_turnup1)),
    dict(call_entry8 = get_date_intersect(df_monthly, stoch_turnup2)),
    dict(call_entry9 = get_date_intersect(df_monthly, rsi_turnup))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

buy_call_backspread = [
    {'C' : [('delta', 0.5, -1), ('delta', 0.25, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 3)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 2)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 3)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35], 
            [42, 70]
             ]

#4. 청산 조건
exit_condition = [
    dict(call_exit1 = []),
    dict(call_exit2 = get_date_intersect(df_monthly, psar_turndown)),
    dict(call_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))),
    dict(call_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [0.25, 0.5, 1, 2, 4]
#6. 손절
stop_loss = [-0.25, -0.5]

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
        res = backtest.get_vertical_trade_result(df_monthly,
                                entry_dates = entry_value,
                                trade_spec = trade,
                                dte_range = dte,
                                exit_dates = exit_value,
                                stop_dte = 1,
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

#%% 콜백스프레드 매수
   
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
    dict(call_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)),
    dict(call_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)),
    dict(call_entry3 = get_date_intersect(df_monthly, psar_turnup)),
    dict(call_entry4 = get_date_intersect(df_monthly, supertrend_turnup)),
    dict(call_entry5 = get_date_intersect(df_monthly, bbands_turnup1)),
    dict(call_entry6 = get_date_intersect(df_monthly, bbands_turnup2)),
    dict(call_entry7 = get_date_intersect(df_monthly, stoch_turnup1)),
    dict(call_entry8 = get_date_intersect(df_monthly, stoch_turnup2)),
    dict(call_entry9 = get_date_intersect(df_monthly, rsi_turnup))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

buy_call_backspread = [
    {'C' : [('delta', 0.5, -1), ('delta', 0.25, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 3)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 2)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 3)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 36], 
            [42, 71]
             ]

#4. 청산 조건
exit_condition = [
    dict(call_exit1 = []),
    dict(call_exit2 = get_date_intersect(df_monthly, psar_turndown)),
    dict(call_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))),
    dict(call_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5))),
    dict(call_exit5 = get_date_union(df_monthly, psar_turndown, bbands_turndown2))
]

#5. 익절 
profit_target = [0.5, 1, 2, 4, 6, 999]
#6. 손절
stop_loss = [-0.5, -1, -2, -3, -999]

comb = list(product(entry_condition, buy_call_backspread, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_call/{i}_{i} + 100.csv")
    del df_res
    del chunk

#%% 콜백스프레드 위클리
   
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
    dict(call_entry10 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trendup)),
    dict(call_entry11 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trendup)),
    dict(call_entry12 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trendup)),
    dict(call_entry13 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trendup)),
    dict(call_entry3 = get_date_intersect(df_weekly, psar_turnup)),
    dict(call_entry4 = get_date_intersect(df_weekly, supertrend_turnup)),
    dict(call_entry5 = get_date_intersect(df_weekly, bbands_turnup1)),
    dict(call_entry6 = get_date_intersect(df_weekly, bbands_turnup2)),
    dict(call_entry7 = get_date_intersect(df_weekly, stoch_turnup1)),
    dict(call_entry8 = get_date_intersect(df_weekly, stoch_turnup2)),
    dict(call_entry9 = get_date_intersect(df_weekly, rsi_turnup))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)
buy_call_backspread = [
    {'C' : [('delta', 0.5, -1), ('delta', 0.25, 2)]},
    {'C' : [('delta', 0.4, -1), ('delta', 0.2, 2)]},
    {'C' : [('delta', 0.3, -1), ('delta', 0.15, 2)]},
    {'C' : [('delta', 0.2, -1), ('delta', 0.1, 2)]}
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
profit_target = [0.5, 1, 2, 4, 999]
#6. 손절
stop_loss = [-0.5, -1, -2, -3, -999]

comb = list(product(entry_condition, buy_call_backspread, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_call/{i}_{i} + 100.csv")
    del df_res
    del chunk


#%% 풋매수

from itertools import product
import time

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


#1. 진입조건
entry_condition = [
    dict(put_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)),
    dict(put_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)),
    dict(put_entry3 = get_date_intersect(df_monthly, psar_turndown)),
    dict(put_entry4 = get_date_intersect(df_monthly, supertrend_turndown)),
    dict(put_entry5 = get_date_intersect(df_monthly, bbands_turndown1)),
    dict(put_entry6 = get_date_intersect(df_monthly, bbands_turndown2)),
    dict(put_entry7 = get_date_intersect(df_monthly, stoch_turndown1)),
    dict(put_entry8 = get_date_intersect(df_monthly, stoch_turndown2)),
    dict(put_entry9 = get_date_intersect(df_monthly, rsi_turndown))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

buy_put =[
    {'P': [('delta', -0.4, 1)]},
    # {'C': [('delta', 0.3, 1)]},
    {'P': [('delta', -0.2, 1)]},
    # {'C': [('delta', 0.1, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35], 
            [42, 70]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = []),
    dict(put_exit2 = get_date_intersect(df_monthly, psar_turnup)),
    dict(put_exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))),
    dict(put_exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [0.25, 0.5, 1, 2, 4]
#6. 손절
stop_loss = [-0.25, -0.5]

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
        res = backtest.get_vertical_trade_result(df_monthly,
                                entry_dates = entry_value,
                                trade_spec = trade,
                                dte_range = dte,
                                exit_dates = exit_value,
                                stop_dte = 1,
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

#%% 풋백스프레드 매수

from itertools import product
import time

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


#1. 진입조건
entry_condition = [
    dict(put_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)),
    dict(put_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)),
    dict(put_entry3 = get_date_intersect(df_monthly, psar_turndown)),
    dict(put_entry4 = get_date_intersect(df_monthly, supertrend_turndown)),
    dict(put_entry5 = get_date_intersect(df_monthly, bbands_turndown1)),
    dict(put_entry6 = get_date_intersect(df_monthly, bbands_turndown2)),
    dict(put_entry7 = get_date_intersect(df_monthly, stoch_turndown1)),
    dict(put_entry8 = get_date_intersect(df_monthly, stoch_turndown2)),
    dict(put_entry9 = get_date_intersect(df_monthly, rsi_turndown))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

buy_put_backspread =[
    {'P': [('delta', -0.5, -1), ('delta', -0.26, 2)]},
    {'P': [('delta', -0.4, -1), ('delta', -0.21, 2)]},
    {'P': [('delta', -0.3, -1), ('delta', -0.16, 2)]},
    {'P': [('delta', -0.2, -1), ('delta', -0.1, 2)]},

]

#3. 어떤 만기 종목
dte_range = [
            [7, 36], 
            [42, 71]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = []),
    dict(put_exit2 = get_date_intersect(df_monthly, psar_turnup)),
    dict(put_exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))),
    dict(put_exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [0.5, 2, 4, 6, 999]
#6. 손절
stop_loss = [-0.5, -1, -2, -3, -999]

comb = list(product(entry_condition, buy_put_backspread, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_put/{i}_{i} + 100.csv")
    del df_res
    del chunk

#%% 풋백스프레드 매수 위클리

from itertools import product
import time

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


#1. 진입조건
entry_condition = [
    dict(put_entry1 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), psar_trenddown)),
    dict(put_entry2 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [3]), supertrend_trenddown)),
    dict(put_entry10 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), psar_trenddown)),
    dict(put_entry11 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [4]), supertrend_trenddown)),
    dict(put_entry12 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), psar_trenddown)),
    dict(put_entry13 = get_date_intersect(df_weekly, weekday_entry(df_weekly, [0]), supertrend_trenddown))
    # dict(put_entry3 = get_date_intersect(df_weekly, psar_turndown)),
    # dict(put_entry4 = get_date_intersect(df_weekly, supertrend_turndown)),
    # dict(put_entry5 = get_date_intersect(df_weekly, bbands_turndown1)),
    # dict(put_entry6 = get_date_intersect(df_weekly, bbands_turndown2)),
    # dict(put_entry7 = get_date_intersect(df_weekly, stoch_turndown1)),
    # dict(put_entry8 = get_date_intersect(df_weekly, stoch_turndown2)),
    # dict(put_entry9 = get_date_intersect(df_weekly, rsi_turndown))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)


buy_put_backspread =[
    {'P': [('delta', -0.5, -1), ('delta', -0.26, 2)]},
    {'P': [('delta', -0.4, -1), ('delta', -0.21, 2)]},
    {'P': [('delta', -0.3, -1), ('delta', -0.16, 2)]},
    {'P': [('delta', -0.2, -1), ('delta', -0.1, 2)]}]

#3. 어떤 만기 종목
dte_range = [
            [2, 9]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = [])
]

#5. 익절 
profit_target = [0.5, 1, 2, 4, 999]
#6. 손절
stop_loss = [-0.5, -1, -2, -3, -999]

comb = list(product(entry_condition, buy_put_backspread, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_buy_put/{i}_{i} + 100.csv")
    del df_res
    del chunk
#%% 풋매도

# 테스트할 전략
    
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
    dict(put_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trendup)),
    dict(put_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trendup)),
    dict(put_entry3 = get_date_intersect(df_monthly, psar_turnup)),
    dict(put_entry4 = get_date_intersect(df_monthly, supertrend_turnup)),
    dict(put_entry5 = get_date_intersect(df_monthly, bbands_turnup1)),
    dict(put_entry6 = get_date_intersect(df_monthly, bbands_turnup2)),
    dict(put_entry7 = get_date_intersect(df_monthly, stoch_turnup1)),
    dict(put_entry8 = get_date_intersect(df_monthly, stoch_turnup2)),
    dict(put_entry9 = get_date_intersect(df_monthly, rsi_turnup))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

sell_put =[
    {'P': [('delta', -0.4, -1)]},
    # {'C': [('delta', 0.3, 1)]},
    {'P': [('delta', -0.2, -1)]},
    # {'C': [('delta', 0.1, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35], 
            [42, 70]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = []),
    dict(put_exit2 = get_date_intersect(df_monthly, psar_turndown)),
    dict(put_exit3 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =5 ,d =3 , smooth_d = 3))),
    dict(put_exit4 = get_date_union(df_monthly, psar_turndown, k200.stoch.rebound1(pos ='s', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [0.25, 0.5, 0.8]
#6. 손절
stop_loss = [-0.25, -0.5, -1, -2]

comb = list(product(entry_condition, sell_put, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_put_short/{i}_{i} + 100.csv")
    del df_res
    del chunk

#%%
# 콜매도
# 테스트할 전략
    
from itertools import product
import time

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


#1. 진입조건
entry_condition = [
    dict(call_entry1 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), psar_trenddown)),
    dict(call_entry2 = get_date_intersect(df_monthly, weekday_entry(df_monthly, [0]), supertrend_trenddown)),
    dict(call_entry3 = get_date_intersect(df_monthly, psar_turndown)),
    dict(call_entry4 = get_date_intersect(df_monthly, supertrend_turndown)),
    dict(call_entry5 = get_date_intersect(df_monthly, bbands_turndown1)),
    dict(call_entry6 = get_date_intersect(df_monthly, bbands_turndown2)),
    dict(call_entry7 = get_date_intersect(df_monthly, stoch_turndown1)),
    dict(call_entry8 = get_date_intersect(df_monthly, stoch_turndown2)),
    dict(call_entry9 = get_date_intersect(df_monthly, rsi_turndown))
]

#2. 전략 선정 (종목 / 행사가 / 수량 / 포지션 선택)

sell_call =[
    {'C': [('delta', 0.4, -1)]},
    # {'C': [('delta', 0.3, 1)]},
    {'C': [('delta', 0.2, -1)]},
    # {'C': [('delta', 0.1, 1)]}
]

#3. 어떤 만기 종목
dte_range = [
            [7, 35], 
            [42, 70]
             ]

#4. 청산 조건
exit_condition = [
    dict(put_exit1 = []),
    dict(put_exit2 = get_date_intersect(df_monthly, psar_turnup)),
    dict(put_exit3 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =5 ,d =3 , smooth_d = 3))),
    dict(put_exit4 = get_date_union(df_monthly, psar_turnup, k200.stoch.rebound1(pos ='l', k =10 ,d =5 , smooth_d = 5)))
]

#5. 익절 
profit_target = [0.25, 0.5, 0.8]
#6. 손절
stop_loss = [-0.25, -0.5, -1, -2]

comb = list(product(entry_condition, sell_call, dte_range, exit_condition, profit_target, stop_loss))

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
    csv_res.to_csv(f"./res_dump_call_short/{i}_{i} + 100.csv")
    del df_res
    del chunk

