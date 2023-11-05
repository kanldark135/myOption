#%%

import pandas as pd
import numpy as np
import option_calc as calc
import compute
from datetime import datetime
import sql
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

def get_summary(df_result):
    res = df_result['summary']
    return res

def get_cumsum(df_result):
    res = df_result['daily_ret'].cumsum()
    return res

def get_sortvalue(df_result):
    res = df_result['all_trades'].sort_values('final_ret', ascending = False)
    return res

# entry_date : 온갖 방법으로 entry date 도출

from get_entry_date import get_date_intersect, get_date_union, weekday_entry, contrarian, notrade

df_k200 = pd.read_pickle("./working_data/df_k200.pkl")

#진입일수 필터링 (날짜)
entry_weekday = weekday_entry(df_k200, [0, 4]) #1. 매주 n요일날 진입

#역발상_상승전환 ~ 상승
entry_stoch_long = df_k200.contra.stoch_rebound(l_or_s = 'l', k = 5, d = 3, smooth_d = 3)
entry_bbands_long = df_k200.contra.through_bbands(l_or_s = 'l', length = 20, std = 2)
entry_rsi_long = df_k200.contra.rsi_rebound(l_or_s = 'l', length = 14, scalar = 20)
entry_psar_long = df_k200.contra.psar_rebound(l_or_s = 'l')

#모멘텀_상승지속??

#역발상_하락전환 ~ 하락
entry_stoch_short = df_k200.contra.stoch_rebound(l_or_s = 's', k = 5, d = 3, smooth_d = 3)
entry_bbands_short = df_k200.contra.through_bbands(l_or_s = 's', length = 20, std = 2)
entry_rsi_short = df_k200.contra.rsi_rebound(l_or_s = 's', length = 14, scalar = 20)
entry_psar_short = df_k200.contra.psar_rebound(l_or_s = 's')

#모멘텀_하락지속??
noentry_stoch_short = flip(entry_stoch_short)
noentry_stoch_long = flip(entry_stoch_long)

#변동성 감소 ~ contained
noentry_stoch_limit = flip(entry_stoch_long.combine_first(entry_stoch_short))
noentry_vix_curve = notrade.no_vix_curve_invert() #3. vix curve not invert & not 하락추세일때 진입
noentry_vkospi_below_n = notrade.no_vkospi_below_n(quantile = 0.2) # vkospi n보다 낮으면 진입 X
noentry_vkospi_above_n = notrade.no_vkospi_above_n(quantile = 0.2) # vkospi n보다 높으면 진입 X

#변동성 증대
entry_vkospi_below_n = flip(notrade.no_vkospi_below_n(0.2))


##풋매도계열 진입 (변동성 높고 / 하방 위험할때 x)------------------

date_sell_put = dict(
date_sell_put1 = get_date_intersect(df_monthly, entry_weekday), # 일만 정해놓고 진입
date_sell_put2 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n), # 일 + X 스토캐스틱 과열
date_sell_put3 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve), # 일 + X 스토캐스틱 과열 + X vkospi200
date_sell_put4 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short), # 일 + X 스토캐스틱 과열 + X vkospi200 +
date_sell_put5 = get_date_intersect(df_monthly), # 일 안정하고 진입
date_sell_put6 = get_date_intersect(df_monthly, noentry_vkospi_below_n), # 일 + X 스토캐스틱 과열
date_sell_put7 = get_date_intersect(df_monthly, noentry_vkospi_below_n, noentry_vix_curve), # 일 + X 스토캐스틱 과열 + X vkospi200
date_sell_put8 = get_date_intersect(df_monthly, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short) # 일 + X 스토캐스틱 과열 + X vkospi200 +
)

#뉴트럴매도 진입 : 변동성 높고 / 위아래 방향성 아닌 경우--------------------------------------------------
date_neutral = dict(
date_neutral1 = get_date_intersect(df_monthly),
date_neutral2 = get_date_intersect(df_monthly, entry_weekday),
date_neutral3 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n),
date_neutral4 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve),
date_neutral5 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_limit),
date_neutral6 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short)
)

#----------------------------------------------------
entry_weekday_w = weekday_entry(df_k200, [3, 4]) # 매주 목/금요일 진입 
date_neutralw = dict(
date_neutralw1 = get_date_intersect(df_weekly),
date_neutralw2 = get_date_intersect(df_weekly, entry_weekday_w),
date_neutralw3 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n),
date_neutralw4 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve),
date_neutralw5 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_limit),
date_neutralw6 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short),
)


#%% finalized quick

entry0 = get_date_intersect(df_monthly)
entry1 = get_date_intersect(df_monthly, entry_psar_long)
entry2 = get_date_union(df_monthly, entry_psar_long, entry_stoch_long)
entry3 = get_date_intersect(df_monthly, entry_psar_long, notrade.no_vkospi_above_n(0.8))
exit = get_date_intersect(df_monthly, entry_psar_short)

# #만기까지 홀딩 안하고 전날 손익불문 강제 청산 반영
# exit_before_expiry = pd.to_datetime((df_monthly['expiry'] - pd.DateOffset(days = 1)).drop_duplicates().values)
# exit = pd.to_datetime(np.sort(np.concatenate([exit, exit_before_expiry])))

dte_range = [35, 70]

quickres1 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = entry0,
                                              trade_spec = sell_call_credit[2],
                                              dte_range = dte_range,
                                              exit_dates = [],
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -4)

#%% long call

#콜매수계열 진입 (변동성 낮고 / --------------------------------------------------
date_buy_call = dict(
alltime = get_date_intersect(df_monthly),
bbands = get_date_intersect(df_monthly, entry_bbands_long),
psar = get_date_intersect(df_monthly, entry_psar_long),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_long, notrade.no_vkospi_above_n(quantile = 0.5)),
rsi = get_date_intersect(df_monthly, entry_rsi_long),
stoch = get_date_intersect(df_monthly, entry_stoch_long)
)

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

# 기본적으로 콜 skew 누워서 불리 > skew 따라 이득보려면 아예 매수부터 외가에 구축해야함


res_call = dict()

for key, values in date_buy_call.items():
    for trade in buy_call:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            # for exit_date in list of exit dates or just declare exit date
            for profit_target in [1, 2, 3, 4]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = False,
                                                    profit_take = profit_target,
                                                    stop_loss = -0.5)
                res_call[f"{key}_{trade}_{dte}_{profit_target}"] = res

res_call_debit = dict()

for key, values in date_buy_call.items():
    for trade in buy_call_debit:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            # for exit_date in list of exit dates or just declare exit date
            for profit_target in [1, 2, 3, 4]:
                res  = backtest.get_vertical_trade_result(df_monthly,   
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = False,
                                                    profit_take = profit_target,
                                                    stop_loss = -0.5)
                res_call_debit[f"{key}_{trade}_{dte}_{profit_target}"] = res

res_call_backspread = dict()

for key, values in date_buy_call.items():
    for trade in buy_call_backspread:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for profit_target in [1, 2, 3, 4, 5]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = profit_target,
                                                    stop_loss = -1.5)
                res_call_backspread[f"{key}_{trade}_{dte}_{profit_target}"] = res

res_call_calendar = dict()

for key, values in date_buy_call.items():
        for profit_target in [1, 2, 3, 4]:
            res  = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = sell_call_front,
                                                    back_spec = buy_call_back,
                                                    front_dte = [14, 35],
                                                    back_dte = [28, 77],
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = profit_target,
                                                    stop_loss = -1)
            res_call_calendar[f"{key}_{dte}_{profit_target}"] = res

#%% short call

# 콜매도계열 진입

date_sell_call = dict(
bbands = get_date_intersect(df_monthly, entry_bbands_short),
bbands_no_highvol = get_date_intersect(df_monthly, entry_bbands_short, notrade.no_vkospi_below_n(0.2)),
psar = get_date_intersect(df_monthly, entry_psar_short),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_short, notrade.no_vkospi_below_n(quantile = 0.2)),
rsi = get_date_intersect(df_monthly, entry_rsi_short),
stoch = get_date_intersect(df_monthly, entry_stoch_short),
stoch_no_highvol = get_date_intersect(df_monthly, entry_stoch_short, notrade.no_vkospi_below_n(0.2))
)

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

res_call = dict()

for key, values in date_sell_call.items():
    for trade in sell_call:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for stop_loss in [-1, -2, -3, -4]:
            # for exit_date in list of exit dates or just declare exit date
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                        entry_dates = values,
                                                        trade_spec = trade,
                                                        dte_range = dte,
                                                        exit_dates = [],
                                                        is_complex_strat = False,
                                                        profit_take = 0.5,
                                                        stop_loss = stop_loss)
                res_call[f"{key}_{trade}_{dte}_{stop_loss}"] = res

res_call_credit = dict()

for key, values in date_sell_call.items():
    for trade in sell_call_credit:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for stop_loss in [-1, -2, -3, -4]:            
            # for exit_date in list of exit dates or just declare exit date
                res  = backtest.get_vertical_trade_result(df_monthly,   
                                                        entry_dates = values,
                                                        trade_spec = trade,
                                                        dte_range = dte,
                                                        exit_dates = [],
                                                        is_complex_strat = False,
                                                        profit_take = 0.5,
                                                        stop_loss = stop_loss)
                res_call_credit[f"{key}_{trade}_{dte}_{stop_loss}"] = res

res_call_ratio = dict()

for key, values in date_sell_call.items():
    for trade in sell_call_ratio:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for stop_loss in [-1, -2, -3, -4]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = 2,
                                                    stop_loss = stop_loss)
                res_call_ratio[f"{key}_{trade}_{dte}_{stop_loss}"] = res


#%%

#풋매수계열 진입 ---------------

date_buy_put = dict(
alltime = get_date_intersect(df_monthly),
bbands = get_date_intersect(df_monthly, entry_bbands_short),
bbands_no_highvol = get_date_intersect(df_monthly, entry_bbands_short, notrade.no_vkospi_above_n(0.8)),
psar = get_date_intersect(df_monthly, entry_psar_short),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_short, notrade.no_vkospi_above_n(quantile = 0.8)),
rsi = get_date_intersect(df_monthly, entry_rsi_short),
stoch = get_date_intersect(df_monthly, entry_stoch_short),
stoch_no_highvol = get_date_intersect(df_monthly, entry_stoch_short, notrade.no_vkospi_above_n(0.8))
)

# long put

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

sell_put_front = {'P': [('delta', -0.4, -1)]}
buy_put_back = {'P': [('delta', -0.2, 2)]} 

res_put = dict()

for key, values in date_buy_put.items():
    for trade in buy_put:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            # for exit_date in list of exit dates or just declare exit date
            for profit_target in [1, 2, 4, 6]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = False,
                                                    profit_take = profit_target,
                                                    stop_loss = -0.5)
                res_put[f"{key}_{trade}_{dte}_{profit_target}"] = res

res_put_debit = dict()

for key, values in date_buy_put.items():
    for trade in buy_put_debit:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            # for exit_date in list of exit dates or just declare exit date
            for profit_target in [0.5, 1, 2, 3]:
                res  = backtest.get_vertical_trade_result(df_monthly,   
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = False,
                                                    profit_take = profit_target,
                                                    stop_loss = -0.5)
                res_put_debit[f"{key}_{trade}_{dte}_{profit_target}"] = res

#%% 
res_put_backspread = dict()

for key, values in date_buy_put.items():
    for trade in buy_put_backspread:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for profit_target in [0.5, 1, 2, 4, 7, 10]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = profit_target,
                                                    stop_loss = -1.5)
                res_put_backspread[f"{key}_{trade}_{dte}_{profit_target}"] = res

res_put_calendar = dict()

for key, values in date_buy_put.items():
        for profit_target in [1, 2, 3, 4]:
            res  = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = sell_call_front,
                                                    back_spec = buy_call_back,
                                                    front_dte = [14, 35],
                                                    back_dte = [28, 77],
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = profit_target,
                                                    stop_loss = -1)
            res_put_calendar[f"{key}_{dte}_{profit_target}"] = res


#%%

# put bwb
sell_put_bwb = {'P' : [('pct', -0.06, 5), ('pct', -0.09, -11), ('pct', -0.15, 5)]}

# strangle
sell_strangle20 = {'C': [('delta', 0.2, -1)], 
                'P': [('delta', -0.2, -1)]}
sell_strangle15 = {'C': [('delta', 0.15, -1)], 
                'P': [('delta', -0.15, -1)]}
sell_strangle10 = {'C': [('delta', 0.10, -1)], 
                'P': [('delta', -0.10, -1)]}
sell_strangle05 = {'C': [('delta', 0.05, -1)], 
                'P': [('delta', -0.05, -1)]}

sell_strangle6pct = {'C': [('pct', 0.06, -1)], 
                'P': [('pct', -0.06, -1)]}
sell_strangle8pct = {'C': [('pct', 0.08, -1)], 
                'P': [('pct', -0.08, -1)]}
sell_strangle10pct = {'C': [('pct', 0.10, -1)], 
                'P': [('pct', -0.10, -1)]}

# weekly
sell_weeklyput75pt = {'P': [('number', -7.5, -1)]}
sell_weeklyput10pt = {'P': [('number', -10, -1)]}
sell_weeklystrangle75pt = {'C': [('number', 7.5, -1)], 
                'P': [('number', -7.5, -1)]}
sell_weeklystrangle10pt = {'C': [('number', 10, -1)], 
                'P': [('number', -10, -1)]}

# put 111
sell_put1111 = {'P': [('delta', -0.25, 1), ('delta', -0.22, -1), ('delta', -0.10, -1)]}
sell_put1112 = {'P': [('delta', -0.25, 1), ('delta', -0.20, -1), ('delta', -0.12, -1)]}

sell_put1121 = {'P': [('delta', -0.25, 1), ('delta', -0.22, -1), ('delta', -0.05, -2)]}
sell_put1122 = {'P': [('delta', -0.25, 1), ('delta', -0.20, -1), ('delta', -0.07, -2)]}

# put calendar series
sell_put40 = {'P': [('delta', -0.40, -1)]}
sell_put30 = {'P': [('delta', -0.30, -1)]}
sell_put25 = {'P': [('delta', -0.25, -1)]}
sell_put20 = {'P': [('delta', -0.20, -1)]}
sell_put15 = {'P': [('delta', -0.15, -1)]}
sell_put10 = {'P': [('delta', -0.10, -1)]}
sell_put05 = {'P': [('delta', -0.05, -1)]}

sell_put_bwb = {'P' : [('pct', -0.06, 5), ('pct', -0.09, -11), ('pct', -0.15, 5)]}

buy_put_front1 = {"P" : [('delta', -0.38, 1)]}
sell_put_back1 = {"P" : [('delta', -0.19, -2)]}

buy_put_front2 = {"P" : [('delta', -0.25, 1), ('delta', -0.22, -1)]}
sell_put_back2 = {"P" : [('delta', -0.1, -1)]}

buy_put_front3 = {"P" : [('delta', -0.25, 1), ('delta', -0.22, -1)]}
sell_put_back3 = {"P" : [('delta', -0.05, -2)]}

buy_put_front4 = {"P" : [('delta', -0.25, 1), ('delta', -0.20, -1)]}
sell_put_back4 = {"P" : [('delta', -0.13, -1)]}

buy_put_front5 = {"P" : [('delta', -0.25, 1), ('delta', -0.20, -1)]}
sell_put_back5 = {"P" : [('delta', -0.07, -2)]}

#%%

exit_psar_short = get_date_intersect(df_monthly, entry_psar_long)

buy_put = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = date_buy_put.get('psar'),
                                                trade_spec = buy_put20,
                                                dte_range = [14, 49],
                                                exit_dates = exit_psar_short,
                                                is_complex_strat = False,
                                                profit_take = 1,
                                                stop_loss = -0.5)

putbackspread = []

for i in buy_put_backspread:

    put_backspread = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = date_buy_put.get('psar'),
                                                    trade_spec = i,
                                                    dte_range = [14, 49],
                                                    exit_dates = exit_psar_short,
                                                    is_complex_strat = True,
                                                    profit_take = 4,
                                                    stop_loss = -1)
    putbackspread.append(put_backspread)


#%%

putcalendar2 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front2,
                                                    back_spec = sell_put_back2,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )         

buy_call = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = date_sell_put.get('date_sell_put4'),
                                                trade_spec = buy_call40,
                                                dte_range = [7, 70],
                                                is_complex_strat = False,
                                                profit_take = 4,
                                                stop_loss = -0.5)

#%% loop_231028 일단 주요 진입조건 변화에 따른 sell_put 전략들 위주로

# 일부 영업일보다 투자횟수가 많게 나오는 이유 : dte_range 상에 근월물 / 차월물이 둘 다 해당되는 경우 특정 진입일날은 근월물 trade / 차월물 trade 둘다 구축
# 이때 이걸 서로 다른 2개의 trade 로 취급
# 이런 매매들이 누적되서 영업일보다 매매횟수가 더 커지는것이므로 이상 없는거임

#%%

sellput30 = []
sellput20 = []
sellput15 = []
sellput10 = []
sellput05 = []

sellputbwb = []

sellput1111 = []
sellput1112 = []
sellput1121 = []
sellput1122 = []

sellputcalendar1 = [] # 근매수 1 / 차매도 2
sellputcalendar2 = [] # 근pds 1 / 차매도 1
sellputcalendar3 = [] # 근pds 1 / 차매도 1
sellputcalendar4 = [] # 근pds 1 / 차매도 1
sellputcalendar5 = [] # 근pds 1 / 차매도 1

for keys, values in date_sell_put.items():

    put20 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put20,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    sellput20.append(put20)

    put15 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put15,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    sellput15.append(put15)

    put10 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put10,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    sellput10.append(put10)

    put05 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put05,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    sellput05.append(put05)
    


    putbwb = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put_bwb,
                                                dte_range = [35 ,70],
                                                is_complex_strat = True,
                                                profit_take = 2,
                                                stop_loss = -4)
    sellputbwb.append(putbwb)
    
    put1111 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put1111,
                                                dte_range = [35, 70],
                                                is_complex_strat = True,
                                                profit_take = 1,
                                                stop_loss = -2.5)
                                            
    sellput1111.append(put1111)
    
    put1112 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put1112,
                                                dte_range = [35, 70],
                                                is_complex_strat = True,
                                                profit_take = 1,
                                                stop_loss = -2.5)
    sellput1112.append(put1112)    

    put1121 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put1121,
                                                dte_range = [35, 70],
                                                is_complex_strat = True,
                                                profit_take = 1,
                                                stop_loss = -2.5)
    sellput1121.append(put1121)  

    put1122 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_put1122,
                                                dte_range = [35, 70],
                                                is_complex_strat = True,
                                                profit_take = 1,
                                                stop_loss = -2.5)
    sellput1122.append(put1122)

    putcalendar1 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front1,
                                                    back_spec = sell_put_back1,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )                                            
    
    sellputcalendar1.append(putcalendar1)
    putcalendar2 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front2,
                                                    back_spec = sell_put_back2,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )                                            
    
    sellputcalendar2.append(putcalendar2)
    putcalendar3 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front3,
                                                    back_spec = sell_put_back3,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )                                            
    
    sellputcalendar3.append(putcalendar3)
    putcalendar4 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front4,
                                                    back_spec = sell_put_back4,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )                                            
    
    sellputcalendar4.append(putcalendar4)
    putcalendar5 = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = buy_put_front5,
                                                    back_spec = sell_put_back5,
                                                    front_dte = [21, 42],
                                                    back_dte = [28, 77],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = -2.5
    )                                            
    
    sellputcalendar5.append(putcalendar5)
#%%


sellstrangle20 = []
sellstrangle15 = []
sellstrangle10 = []
sellstrangle05 = []
sellstrangle6pct = []
sellstrangle8pct = []
sellstrangle10pct = []


for keys, values in date_neutral.items():

    strangle20 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle20,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle20.append(strangle20)
    strangle15 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle15,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle15.append(strangle15)
    strangle10 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle10,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle10.append(strangle10)
    strangle05 = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle05,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle05.append(strangle05)
    strangle6pct = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle6pct,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle6pct.append(strangle6pct)
    strangle8pct = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle8pct,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle8pct.append(strangle8pct)
    strangle10pct = backtest.get_vertical_trade_result(df_monthly,
                                                entry_dates = values,
                                                trade_spec = sell_strangle10pct,
                                                dte_range = [35, 70],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellstrangle10pct.append(strangle10pct)

# 아예 전략 두개 합성 후 손익관리
#%% 

leg_pds = backtest.get_vertical_trade_result(df_monthly,
                                               entry_dates = date_sell_put['date_sell_put3'],
                                               trade_spec = {"P" : [('delta', -0.25, 1), ('delta', -0.22, -1)]},
                                               dte_range = [35, 70],
                                               is_complex_strat = True,
                                               profit_take = 2.45,
                                               stop_loss = -1000)
leg_np = backtest.get_vertical_trade_result(df_monthly,
                                               entry_dates = date_sell_put['date_sell_put3'],
                                               trade_spec = {"P" : [('delta', -0.05, -2)]},
                                               dte_range = [35, 70],
                                               is_complex_strat = False,
                                               profit_take = 0.95,
                                               stop_loss = -4)


#%%

sellweeklyput10pt = []
sellweeklyput75pt = []
sellweeklystrangle10pt = []
sellweeklystrangle75pt = []

df_weekly = df_weekly.loc[~df_weekly['iv'].isna()]

for keys, values in date_neutralw.items():

    weeklyput75pt = backtest.get_vertical_trade_result(df_weekly,
                                                entry_dates = values,
                                                trade_spec = sell_weeklyput75pt,
                                                dte_range = [6, 9],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellweeklyput75pt.append(weeklyput75pt)   

    weeklyput10pt = backtest.get_vertical_trade_result(df_weekly,
                                                entry_dates = values,
                                                trade_spec = sell_weeklyput10pt,
                                                dte_range = [6, 9],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellweeklyput10pt.append(weeklyput10pt)   

    weeklystrangle75pt = backtest.get_vertical_trade_result(df_weekly,
                                                entry_dates = values,
                                                trade_spec = sell_weeklystrangle75pt,
                                                dte_range = [6, 9],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellweeklystrangle75pt.append(weeklystrangle75pt)   

    weeklystrangle10pt = backtest.get_vertical_trade_result(df_weekly,
                                                entry_dates = values,
                                                trade_spec = sell_weeklystrangle10pt,
                                                dte_range = [6, 9],
                                                is_complex_strat = False,
                                                profit_take = 0.5,
                                                stop_loss = -2)
    
    sellweeklystrangle10pt.append(weeklystrangle10pt)   


#%% 
# # 테스트용 예시 : 2023-07-13

# grouped = df_monthly.groupby('expiry')
# all_expiry = grouped.groups.keys()

# dte_range = [35, 70]

# is_complex_strat = True
# profit_take = 2
# stop_loss = -4

# sample = grouped.get_group('2020-03-12')
# sample_pivoted = sample.pipe(backtest.get_pivot_table)

# trade_entry = backtest.create_trade_entries(df_pivoted = sample_pivoted, 
#                                     entry_dates = long_dates, 
#                                     trade_spec = buy_put_bwb,
#                                     dte_range = dte_range)

# trade_res = list(map(lambda trade : backtest.get_single_trade_result(sample_pivoted, trade), trade_entry))
# trade_res_stopped = list(map(lambda result : backtest.stop_single_trade(result, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss), trade_res))

# ret = backtest.get_single_expiry_result(df_pivoted = sample_pivoted, 
#                     entry_dates = long_dates, 
#                     trade_spec = buy_put_bwb,
#                     dte_range = dte_range,                        
#                     is_complex_strat = is_complex_strat, 
#                     profit_take = profit_take, 
#                     stop_loss = stop_loss)

# result = backtest.get_vertical_trade_result(sample,
#                     entry_dates = long_dates,
#                     trade_spec = buy_put_bwb,
#                     dte_range = dte_range,                        
#                     is_complex_strat = is_complex_strat, 
#                     profit_take = profit_take, 
#                     stop_loss = stop_loss
#                     )

# result_calendar = backtest.get_calendar_trade_result(sample,
#                     entry_dates = long_dates,
#                     front_spec = buy_put_front,
#                     back_spec = sell_put_back,
#                     front_dte = [14, 35],
#                     back_dte = [28, 77],
#                     is_complex_strat = is_complex_strat, 
#                     profit_take = profit_take, 
#                     stop_loss = stop_loss
#                     )

# #%%

# long_delta_strat = {
# 'buy_call' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = long_dates, 
#                         trade_spec = buy_call,
#                         dte_range = dte_range,
#                         is_complex_strat = False, 
#                         profit_take = 2, 
#                         stop_loss = -0.5),

# 'sell_put' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = long_dates, 
#                         trade_spec = sell_put,
#                         dte_range = dte_range,
#                         is_complex_strat = False, 
#                         profit_take = 0.5, 
#                         stop_loss = -2),
# }

# #%%

# short_delta_strat = {
# 'sell_call' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = short_dates, 
#                         trade_spec = sell_call,
#                         dte_range = dte_range,
#                         is_complex_strat = False, 
#                         profit_take = 0.5, 
#                         stop_loss = -2),

# 'sell_put' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = short_dates, 
#                         trade_spec = buy_put,
#                         dte_range = dte_range,
#                         is_complex_strat = False, 
#                         profit_take = 2, 
#                         stop_loss = -0.5),
# }
# #%% 

# theta_strat = {
# 'sell_strangle' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = neutral_dates, 
#                         trade_spec = sell_strangle,
#                         dte_range = dte_range,
#                         is_complex_strat = False, 
#                         profit_take = 0.5, 
#                         stop_loss = -2),

# 'sell_ic' : backtest.get_vertical_trade_result(
#                         df = df_monthly,
#                         entry_dates = neutral_dates, 
#                         trade_spec = sell_ic,
#                         dte_range = dte_range,
#                         is_complex_strat = False,
#                         profit_take = 0.5, 
#                         stop_loss = -2),

# 'buy_put111' : backtest.get_vertical_trade_result(
#                         entry_dates = short_dates, 
#                         trade_spec = buy_put111,
#                         dte_range = dte_range,                        
#                         is_complex_strat = True, 
#                         profit_take = 2.5,
#                         stop_loss = -2.5),

# 'buy_calendarized_12' : backtest.get_calendar_trade_result(
#                         entry_dates = short_dates,
#                         front_spec = buy_put,
#                         back_spec = sell_put_calendar,
#                         front_dte = [21, 42],
#                         back_dte = [28, 77],
#                         is_complex_strat = False
#                         profit_take = 1,
#                         stop_loss = 1)
# }