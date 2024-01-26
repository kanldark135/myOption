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
    return res

def cum(df_result):
    res = df_result['daily_ret'].groupby(level = 1).sum().cumsum()
    res.plot()
    return res

def dd(df_result):
    res = cum(df_result) - cum(df_result).cummax()
    res.plot()
    return res

def table(df_result):
    res = df_result['all_trades'].sort_values('final_ret', ascending = False)
    return res

def returns(df_result):
    res = df_result['daily_ret']
    res.to_csv("./daily_ret.csv")

def all(df_result):
    returns(df_result)
    return sum(df_result), table(df_result), print(cum(df_result))

def vol_based_sizing(vkospi, multiplier = 1, vol_percentile = [0.5, 0.75]):
    vol_bins = [0] + vol_percentile + [1]
    trade_volume = [1 + i * multiplier for i in range(len(vol_bins) - 1)]
    vol_rank = vkospi['close'].rank(pct= True)
    res = pd.cut(vol_rank, bins = vol_bins, labels = trade_volume)
    res = res.astype('int64')
    return res

def sized_cum(df_result, df_sizing):
    a = (df_result['all_trades']['trade_ret'] * df_sizing).dropna()
    aa = pd.concat(a.tolist(), axis = 1, ignore_index = True)
    aaa = aa.sum(axis = 1)
    res = aaa.cumsum()
    return res

# entry_date : 온갖 방법으로 entry date 도출

from get_entry_date import get_date_intersect, get_date_union, weekday_entry, MyContrarian, notrade, MyTrend

#역발상_상승전환 ~ 상승
entry_stoch_long = k200.stoch.rebound1(pos = 'l', k = 5, d = 3, smooth_d = 3)
entry_bbands_long = k200.contra.through_bbands(pos = 'l', length = 20, std = 2)
entry_rsi_long = k200.contra.rsi_rebound(pos = 'l', length = 14, scalar = 20)
entry_psar_long = k200.contra.psar_rebound(pos = 'l')

#모멘텀_상승지속??

trend_psar_long = k200.trend.psar_trend(pos = 'l')
trend_stoch_long = k200.stoch.is_updown(pos = 'l', k_or_d = 'k', k = 5, d = 3, smooth_d = 3)

#역발상_하락전환 ~ 하락
entry_stoch_short = k200.stoch.rebound1(pos = 's', k = 5, d = 3, smooth_d = 3)
entry_bbands_short = k200.contra.through_bbands(pos = 's', length = 20, std = 2)
entry_rsi_short = k200.contra.rsi_rebound(pos = 's', length = 14, scalar = 20)
entry_psar_short = k200.contra.psar_rebound(pos = 's')

#모멘텀_하락지속??
noentry_stoch_short = flip(entry_stoch_short)
noentry_stoch_long = flip(entry_stoch_long)
trend_psar_short = k200.trend.psar_trend(pos = 's')
trend_stoch_short = k200.stoch.is_updown(pos = 's', k_or_d = 'k', k = 5, d = 3, smooth_d = 3)

#변동성 감소 ~ contained
noentry_stoch_limit = flip(entry_stoch_long.combine_first(entry_stoch_short))
noentry_vix_curve = notrade.vix_curve_invert() #3. vix curve not invert & not 하락추세일때 진입
noentry_vkospi_below_n = notrade.vkospi_below_n(quantile = 0.2) # vkospi n보다 낮으면 진입 X
noentry_vkospi_above_n = notrade.vkospi_above_n(quantile = 0.2) # vkospi n보다 높으면 진입 X

#변동성 증대
entry_vkospi_below_n = flip(notrade.vkospi_below_n(0.2))


#콜매수계열 진입 (변동성 낮고 / --------------------------------------------------
date_buy_call = dict(
bbands = get_date_intersect(df_monthly, entry_bbands_long),
psar = get_date_intersect(df_monthly, entry_psar_long),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_long, notrade.vkospi_above_n(quantile = 0.5)),
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
# 콜매도계열 진입

date_sell_call = dict(
alltime = get_date_intersect(df_monthly),
bbands = get_date_intersect(df_monthly, entry_bbands_short),
bbands_no_highvol = get_date_intersect(df_monthly, entry_bbands_short, notrade.vkospi_below_n(0.2)),
psar = get_date_intersect(df_monthly, entry_psar_short),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_short, notrade.vkospi_below_n(quantile = 0.2)),
rsi = get_date_intersect(df_monthly, entry_rsi_short),
stoch = get_date_intersect(df_monthly, entry_stoch_short),
stoch_no_highvol = get_date_intersect(df_monthly, entry_stoch_short, notrade.vkospi_below_n(0.2))
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

sell_call_111 = [
    {'C' : [('delta', 0.5, 1), ('delta', 0.46, -1), ('delta', 0.15, -1)]},
    {'C' : [('delta', 0.4, 1), ('delta', 0.36, -1), ('delta', 0.125, -1)]},
    {'C' : [('delta', 0.3, 1), ('delta', 0.26, -1), ('delta', 0.10, -1)]},
    {'C' : [('delta', 0.25, 1), ('delta', 0.20, -1), ('delta', 0.08, -1)]},
    {'C' : [('delta', 0.2, 1), ('delta', 0.15, -1), ('delta', 0.07, -1)]}
]

#풋매수계열 진입 ---------------

date_buy_put = dict(
alltime = get_date_intersect(df_monthly),
lowvol = get_date_intersect(df_monthly, flip(notrade.vkospi_below_n(0.2))),
bbands = get_date_intersect(df_monthly, entry_bbands_short),
bbands_no_highvol = get_date_intersect(df_monthly, entry_bbands_short, notrade.vkospi_above_n(0.8)),
psar = get_date_intersect(df_monthly, entry_psar_short),
psar_no_highvol = get_date_intersect(df_monthly, entry_psar_short, notrade.vkospi_above_n(0.8)),
rsi = get_date_intersect(df_monthly, entry_rsi_short),
stoch = get_date_intersect(df_monthly, entry_stoch_short),
stoch_no_highvol = get_date_intersect(df_monthly, entry_stoch_short, notrade.vkospi_above_n(0.8))
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

date_sell_put = dict(
weekday = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4])),
weekday_no_lowvol = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), notrade.vkospi_below_n(0.2)),
bbands = get_date_intersect(df_monthly, entry_bbands_long),
bbands_no_lowvol = get_date_intersect(df_monthly, entry_bbands_long, notrade.vkospi_below_n(0.2)),
psar = get_date_intersect(df_monthly, entry_psar_long),
psar_no_lowvol = get_date_intersect(df_monthly, entry_psar_long, notrade.vkospi_below_n(0.2)),
rsi = get_date_intersect(df_monthly, entry_rsi_long),
stoch = get_date_intersect(df_monthly, entry_stoch_long),
stoch_no_lowvol = get_date_intersect(df_monthly, entry_stoch_long, notrade.vkospi_below_n(0.2))
)

# short put

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

sell_put_backspread =[
    {'P': [('delta', -0.5, -1), ('delta', -0.26, 2)]},
    {'P': [('delta', -0.4, -1), ('delta', -0.21, 2)]},
    {'P': [('delta', -0.3, -1), ('delta', -0.16, 2)]},
    {'P': [('number', 0, -1), ('number', -5, 2)]},
    {'P': [('number', 0, -1), ('number', -7.5, 2)]}
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



# #뉴트럴매도 진입 : 변동성 높고 / 위아래 방향성 아닌 경우--------------------------------------------------
# date_neutral = dict(
# date_neutral1 = get_date_intersect(df_monthly),
# date_neutral2 = get_date_intersect(df_monthly, entry_weekday),
# date_neutral3 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n),
# date_neutral4 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve),
# date_neutral5 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_limit),
# date_neutral6 = get_date_intersect(df_monthly, entry_weekday, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short)
# )

# #----------------------------------------------------
# entry_weekday_w = weekday_entry(df_k200, [3, 4]) # 매주 목/금요일 진입 
# date_neutralw = dict(
# date_neutralw1 = get_date_intersect(df_weekly),
# date_neutralw2 = get_date_intersect(df_weekly, entry_weekday_w),
# date_neutralw3 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n),
# date_neutralw4 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve),
# date_neutralw5 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_limit),
# date_neutralw6 = get_date_intersect(df_weekly, entry_weekday_w, noentry_vkospi_below_n, noentry_vix_curve, noentry_stoch_short),
# )


#%% finalized quick

put_entry0 = get_date_intersect(df_monthly)
put_entry1 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]))
put_entry2 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_long)
put_entry3 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_long, notrade.vkospi_below_n(0.2))
put_entry4 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_long, notrade.vkospi_below_n(0.2), notrade.vix_curve_invert())
put_entry5 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_long, trend_stoch_long)

#stoch : 과매수상태에서 + 하락하는 경우 1) 첫 하락에서 청산 / 2) 과매수 + 하락상태에서는 진입 X
stoch_overbought = k200.stoch.is_overtraded(pos = 'l') # 과매수
stoch_going_down = k200.stoch.is_updown(pos = 's') # 하락

#psar : 하락 반전 : 첫 하락에서 청산 (고정 청산 시그널)
psar_going_up = k200.trend.psar_trend(pos = 'l')
psar_turn_down = k200.contra.psar_rebound(pos = 's')


put_entry_stoch = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), flip(stoch_overbought * stoch_going_down))
put_entry_comp = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), psar_going_up, flip(stoch_overbought * stoch_going_down))

put_exit_stoch = get_date_union(df_monthly, stoch_overbought * stoch_going_down)
put_exit_comp = get_date_union(df_monthly, stoch_overbought * stoch_going_down, psar_turn_down)

put_exit = get_date_intersect(df_monthly, psar_turn_down)
put_stop = 1

# put_exit = []

dte_range = [42, 70]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = put_entry_comp,
                                              trade_spec = sell_put[0],
                                              dte_range = dte_range,
                                              exit_dates = put_exit_stoch,
                                              stop_dte = put_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.8,
                                              stop_loss = -1)

res_psar_up = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = put_entry2,
                                              trade_spec = sell_put[0],
                                              dte_range = dte_range,
                                              exit_dates = put_exit,
                                              stop_dte = put_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.8,
                                              stop_loss = -1)

# res_psar_up_novix20 = backtest.get_vertical_trade_result(df_monthly,
#                                               entry_dates = put_entry3,
#                                               trade_spec = sell_put[0],
#                                               dte_range = dte_range,
#                                               exit_dates = put_exit,
#                                               stop_dte = put_stop,
#                                               is_complex_strat = False,
#                                               profit_take = 0.5,
#                                               stop_loss = -1)

# res_psar_up_novix20_noinvert = backtest.get_vertical_trade_result(df_monthly,
#                                               entry_dates = put_entry4,
#                                               trade_spec = sell_put[0],
#                                               dte_range = dte_range,
#                                               exit_dates = put_exit,
#                                               stop_dte = put_stop,
#                                               is_complex_strat = False,
#                                               profit_take = 0.5,
#                                               stop_loss = -1)

# res_psar_stoch = backtest.get_vertical_trade_result(df_monthly,
#                                               entry_dates = put_entry5,
#                                               trade_spec = sell_put[0],
#                                               dte_range = dte_range,
#                                               exit_dates = put_exit,
#                                               stop_dte = put_stop,
#                                               is_complex_strat = False,
#                                               profit_take = 0.5,
#                                               stop_loss = -1)


# res_calendar  = backtest.get_calendar_trade_result(df_monthly,
#                                         entry_dates = get_date_intersect(df_monthly, notrade.no_vkospi_below_n(0.2), weekday_entry(df_k200, [0, 4])),
#                                         front_spec = buy_put_front,
#                                         back_spec = sell_put_back,
#                                         front_dte = [14, 35],
#                                         back_dte = [28, 77],
#                                         exit_dates = [],
#                                         is_complex_strat = True,
#                                         profit_take = 1,
#                                         stop_loss = -2)

#%% BSH


entry0 = get_date_intersect(df_monthly)
bsh_entry1 = get_date_intersect(df_monthly, trend_psar_short, weekday_entry(df_monthly, [0, 4]))
bsh_entry2 = get_date_intersect(df_monthly, entry_psar_short, notrade.vkospi_above_n(0.8))
bsh_exit = get_date_intersect(df_monthly, k200.contra.psar_rebound(pos = 's'))

# put_exit = []

# trade = {"P" : [('delta', -0.3, -1), ('delta', -0.1, 1), ('delta', -0.03, 10)]}
trade = {"P" : [('number', 0, -1), ('number', -7.5, 2)]}
dte_range = [42, 70]
stop_bsh = 14

res_bsh = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = bsh_entry2,
                                              trade_spec = trade,
                                              dte_range = dte_range,
                                              exit_dates = [],
                                              stop_dte = stop_bsh,
                                              is_complex_strat = True,
                                              profit_take = 10,
                                              stop_loss = -2)


#%% finalized quick _addhoc for call

entry0 = get_date_intersect(df_monthly)
call_entry1 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]))
call_entry2 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_short)
call_entry3 = get_date_intersect(df_monthly, weekday_entry(k200, [0, 4]), trend_psar_short, notrade.vkospi_above_n(0.8))
call_entry4 = get_date_intersect(df_monthly, entry_psar_short)
call_entry5 = get_date_intersect(df_monthly, entry_psar_short, notrade.vkospi_above_n(0.8))

call_exit = [get_date_intersect(df_monthly, k200.contra.psar_rebound(pos = 'l'))]

call_stop = 1

# call_exit = []

dte_range = [42, 70]

res1 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry1,
                                              trade_spec = sell_call[0],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -1)

res2 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry2,
                                              trade_spec = sell_call[0],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -1)

res3 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry3,
                                              trade_spec = sell_call[2],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -1)

res4 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry4,
                                              trade_spec = sell_call[2],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -1)

res5 = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry5,
                                              trade_spec = sell_call[2],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 0.5,
                                              stop_loss = -1)




#%% finalized quick _addhoc for call

entry0 = get_date_intersect(df_monthly)
call_entry = get_date_intersect(df_monthly, entry_psar_long)
call_stop = 1

# call_exit = []

dte_range = [7, 35]

res = backtest.get_vertical_trade_result(df_monthly,
                                              entry_dates = call_entry,
                                              trade_spec = buy_call[0],
                                              dte_range = dte_range,
                                              exit_dates = call_exit,
                                              stop_dte = call_stop,
                                              is_complex_strat = False,
                                              profit_take = 4,
                                              stop_loss = -0.5)



#%% 

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


#%%
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
res_call_111 = dict()

for key, values in date_sell_call.items():
    for trade in sell_call_111:
        for dte in [[7, 35], [21, 49], [42, 70]]:
            for stop_loss in [-1, -2, -3, -4]:
                res  = backtest.get_vertical_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    trade_spec = trade,
                                                    dte_range = dte,
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = 1,
                                                    stop_loss = stop_loss)
                res_call_111[f"{key}_{trade}_{dte}_{stop_loss}"] = res   

#%%
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

#%% 
res_put_calendar = dict()

for key, values in date_buy_put.items():
        for profit_target in [1, 2, 3, 4]:
            res  = backtest.get_calendar_trade_result(df_monthly,
                                                    entry_dates = values,
                                                    front_spec = sell_put_front,
                                                    back_spec = buy_put_back,
                                                    front_dte = [14, 35],
                                                    back_dte = [28, 77],
                                                    exit_dates = [],
                                                    is_complex_strat = True,
                                                    profit_take = profit_target,
                                                    stop_loss = -1)
            res_put_calendar[f"{key}_{dte}_{profit_target}"] = res


#%% loop_231028 일단 주요 진입조건 변화에 따른 sell_put 전략들 위주로

# 일부 영업일보다 투자횟수가 많게 나오는 이유 : dte_range 상에 근월물 / 차월물이 둘 다 해당되는 경우 특정 진입일날은 근월물 trade / 차월물 trade 둘다 구축
# 이때 이걸 서로 다른 2개의 trade 로 취급
# 이런 매매들이 누적되서 영업일보다 매매횟수가 더 커지는것이므로 이상 없는거임

#%%



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