#%%

import backtest
import pandas as pd

df_monthly = pd.read_pickle("./working_data/df_monthly.pkl")
df_weekly = pd.read_pickle("./working_data/df_weekly.pkl")

data_from = '2010-01-01' # 옛날에는 행사가가 별로 없어서 전략이 이상하게 나감

df_monthly = df_monthly.loc[data_from:]
df_weekly = df_weekly.loc['2019-01-01':]

#%% 
# entry_date : 온갖 방법으로 entry date 도출

from get_entry_date import get_date, weekday_entry, contrarian, notrade

df_k200 = pd.read_pickle("./working_data/df_k200.pkl")

entry_stoch_long = df_k200.my_contrarian.stoch_rebound(k = 5, d = 3, smooth_d = 3, long_or_short = 'l') # 1. stochastic 역발상 매수일때 진입
entry_stoch_short = df_k200.my_contrarian.stoch_rebound(k = 5, d = 3, smooth_d = 3, long_or_short = 's')

entry_bbands_long = df_k200.my_contrarian.through_bbands(20, 2, long_or_short = 'l')
entry_bbands_short = df_k200.my_contrarian.through_bbands(20, 2, long_or_short = 's')

entry_weekday = weekday_entry(df_k200, [0, 4]) #2. 매주 n요일날 진입

entry_vix_curve = notrade.vix_curve_invert() #3. vix curve not invert & not 하락추세일때 진입
entry_vkospi_below_n = notrade.vkospi_below_n(quantile = 0.2) # vkospi n보다 낮으면 진입 X
entry_vkospi_above_n = notrade.vkospi_above_n(quantile = 0.2) # vkospi n보다 높으면 진입 X

##------------------

long_dates = get_date(df_monthly, entry_vkospi_above_n)
short_dates = get_date(df_monthly, entry_bbands_short* -1)

#----------------------------------------------------
dates_w = get_date(df_weekly, entry_weekday, entry_vkospi_below_n)

#--------------------------------------------------

sell_put_weekly = {"P" : [('number', -10, -1)]}
sell_strangle_weekly = {'C': [('number', 10, -1)], 
                'P': [('number', -10, -1)]}

# long delta strategy

buy_call = {'C': [('delta', 0.4, 1)]}
buy_call_cs = {'C' : [('number', 0, 1), ('number', 2.5, -1)]}
sell_put = {'P': [('delta', -0.2, -1)]}

buy_put = {'P': [('delta', -0.4, 1)]}
sell_call = {'C': [('delta', 0.2, -1)]}

# naked call
buy_put_calendar = {'P': [('delta', -0.25, 2)]}
sell_put_calendar = {'P': [('delta', -0.5, -1)]}


# put bwb
buy_put_bwb = {'P' : [('pct', -0.06, 5), ('pct', -0.09, -11), ('pct', -0.15, 5)]}

# naked put
# strangle
sell_strangle = {'C': [('pct', 0.08, -1)], 
                'P': [('pct', -0.08, -1)]}
# iron condor
sell_ic = {'C': [('pct', 0.08, -1), ('pct', 0.09, 1)], 
                'P': [('pct', -0.08, -1), ('pct', -0.09, 1)]}
# put 111
buy_put111 = {'P': [('delta', -0.25, 1), ('delta', -0.21, -1), ('delta', -0.05, -1)]}


#%% 지금 하는 backtest

otm_calendar = backtest.get_calendar_trade_result(df_monthly,
                                                   long_dates,
                                                   front_spec=buy_put_calendar,
                                                   back_spec=sell_put_calendar,
                                                   front_dte = [14, 42],
                                                   back_dte = [28, 77],
                                                   is_complex_strat=False,
                                                   profit_take = 2,
                                                   stop_loss = -0.5)
#%% 
bwb = backtest.get_vertical_trade_result(df_monthly,
                                         long_dates,
                                         buy_put_bwb,
                                         dte_range = [35, 70],
                                         is_complex_strat=True,
                                         profit_take = 2,
                                         stop_loss = -4)

#%% 

result_sell_put_weekly = backtest.get_vertical_trade_result(df_weekly,
                                         dates_w,
                                         sell_put_weekly,
                                         dte_range = [0, 8],
                                         is_complex_strat=False,
                                         profit_take = 0.5,
                                         stop_loss = -2)

result_sell_strangle = backtest.get_vertical_trade_result(df_weekly,
                                         dates_w,
                                         sell_strangle_weekly,
                                         dte_range = [0, 8],
                                         is_complex_strat=False,
                                         profit_take = 0.5,
                                         stop_loss = -2)

#%% 
# 테스트용 예시 : 2023-07-13

grouped = df_monthly.loc[data_from:].groupby('expiry')
all_expiry = grouped.groups.keys()

dte_range = [35, 70]

is_complex_strat = True
profit_take = 2
stop_loss = -4

sample = grouped.get_group('2020-03-12')
sample_pivoted = sample.pipe(backtest.get_pivot_table)

trade_entry = backtest.create_trade_entries(df_pivoted = sample_pivoted, 
                                    entry_dates = long_dates, 
                                    trade_spec = buy_put_bwb,
                                    dte_range = dte_range)

trade_res = list(map(lambda trade : backtest.get_single_trade_result(sample_pivoted, trade), trade_entry))
trade_res_stopped = list(map(lambda result : backtest.stop_single_trade(result, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss), trade_res))

ret = backtest.get_single_expiry_result(df_pivoted = sample_pivoted, 
                    entry_dates = long_dates, 
                    trade_spec = buy_put_bwb,
                    dte_range = dte_range,                        
                    is_complex_strat = is_complex_strat, 
                    profit_take = profit_take, 
                    stop_loss = stop_loss)

result = backtest.get_vertical_trade_result(sample,
                    entry_dates = long_dates,
                    trade_spec = buy_put_bwb,
                    dte_range = dte_range,                        
                    is_complex_strat = is_complex_strat, 
                    profit_take = profit_take, 
                    stop_loss = stop_loss
                    )

result_calendar = backtest.get_calendar_trade_result(sample,
                    entry_dates = long_dates,
                    front_spec = buy_put,
                    back_spec = sell_put_calendar,
                    front_dte = [14, 35],
                    back_dte = [28, 77],
                    is_complex_strat = is_complex_strat, 
                    profit_take = profit_take, 
                    stop_loss = stop_loss
                    )
#%%

long_delta_strat = {
'buy_call' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = long_dates, 
                        trade_spec = buy_call,
                        dte_range = dte_range,
                        is_complex_strat = False, 
                        profit_take = 2, 
                        stop_loss = -0.5),

'sell_put' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = long_dates, 
                        trade_spec = sell_put,
                        dte_range = dte_range,
                        is_complex_strat = False, 
                        profit_take = 0.5, 
                        stop_loss = -2),
}

#%%

short_delta_strat = {
'sell_call' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = short_dates, 
                        trade_spec = sell_call,
                        dte_range = dte_range,
                        is_complex_strat = False, 
                        profit_take = 0.5, 
                        stop_loss = -2),

'sell_put' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = short_dates, 
                        trade_spec = buy_put,
                        dte_range = dte_range,
                        is_complex_strat = False, 
                        profit_take = 2, 
                        stop_loss = -0.5),
}
#%% 

theta_strat = {
'sell_strangle' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = neutral_dates, 
                        trade_spec = sell_strangle,
                        dte_range = dte_range,
                        is_complex_strat = False, 
                        profit_take = 0.5, 
                        stop_loss = -2),

'sell_ic' : backtest.get_vertical_trade_result(
                        df = df_monthly,
                        entry_dates = neutral_dates, 
                        trade_spec = sell_ic,
                        dte_range = dte_range,
                        is_complex_strat = False,
                        profit_take = 0.5, 
                        stop_loss = -2),

'buy_put111' : backtest.get_vertical_trade_result(
                        entry_dates = short_dates, 
                        trade_spec = buy_put111,
                        dte_range = dte_range,                        
                        is_complex_strat = True, 
                        profit_take = 2.5,
                        stop_loss = -2.5),

'buy_calendarized_12' : backtest.get_calendar_trade_result(
                        entry_dates = short_dates,
                        front_spec = buy_put,
                        back_spec = sell_put_calendar,
                        front_dte = [21, 42],
                        back_dte = [28, 77],
                        is_complex_strat = False
                        profit_take = 1,
                        stop_loss = 1)
}

#%%  1. 매매날짜에 따른 차이 + vix curve invert 시 매매 X

res_list_2 = []

for i in range(5):

    entry_weekday = weekday_entry(df_k200, [i]) #2. 매주 n요일날 진입
    entry_vix_curve = notrade.vix_curve_invert() #3. vix curve not invert & not 하락추세일때 진입
    long_dates = get_date(df_monthly, entry_weekday)

    # m_buy_call = grouped.apply(get_final_result, 
    #                         entry_dates = long_dates, 
    #                         trade_spec = buy_call,
    #                         dte_range = dte_range,
    #                         is_complex_strat = False, 
    #                         profit_take = 1, 
    #                         stop_loss = -0.5)

    m_sell_put = grouped.apply(get_vertical_trade_result, 
                            entry_dates = long_dates,
                            dte_range = dte_range,                        
                            trade_spec = sell_put,
                            is_complex_strat = False, 
                            profit_take = 0.5, 
                            stop_loss = -2)
    
    res_list_2.append(m_sell_put)
