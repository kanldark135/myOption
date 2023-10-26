df = df_monthly
front_trade = buy_put
back_trade = sell_put_calendar
all_expiry = grouped.groups.keys()

# 근월 / 차월 만기 pair 생성하는 함수
def get_pair_expiry(all_expiry):
    all_expiry = list(all_expiry)
    res = list(zip(all_expiry, all_expiry[1:]))
    return res

#2. 근월물 진입시점에 맞춰서 차월물 진입시점 일치시키는 함수
def get_filtered_trades_back(trade_front, trade_back):
    front_dates = list(map(lambda trade : trade['entry_date'], trade_front))
    filtered_trades = list(filter(lambda trade : trade['entry_date'] in front_dates, trade_back))
    return filtered_trades

paired_expiry = get_pair_expiry(all_expiry)

# def get_calendar_trade_result(grouped, front_spec, back_spec, entry_dates):

for front, back in paired_expiry:
    df_front = grouped.get_group(front)
    df_back = grouped.get_group(back)

    pivoted_front = get_pivot_table(df_front)
    pivoted_back = get_pivot_table(df_back)

    front_trades = create_trade_entries(pivoted_front, entry_dates = long_dates, trade_spec = buy_put, dte_range = [14, 35])
    back_trades = create_trade_entries(pivoted_back, entry_dates = long_dates, trade_spec = sell_put_calendar, dte_range = [28, 77])
    filtered_back_trades = get_filtered_trades_back(front_trades, back_trades)

    front_trade_res = list(map(lambda trade : get_single_trade_result(pivoted_front, trade), front_trades))
    back_trade_res = list(map(lambda trade : get_single_trade_result(pivoted_back, trade), filtered_back_trades))

    def result_aggregate(front_trade_res, back_trade_res):
        res_list = []
        for i in range(len(front_trade_res)):
            agg_area = pd.concat([front_trade_res[i]['area'], back_trade_res[i]['area']], axis = 1)
            agg_premium = pd.concat([front_trade_res[i]['df_premium'], back_trade_res[i]['df_premium']], axis = 1)
            agg_ret = pd.concat([front_trade_res[i]['df_ret'], back_trade_res[i]['df_ret']], axis = 1)
            agg_cum_ret = pd.concat([front_trade_res[i]['df_cumret'], back_trade_res[i]['df_cumret']], axis = 1)
            agg_daily_ret = front_trade_res[i]['daily_ret'] + back_trade_res[i]['daily_ret']
            agg_cumret = front_trade_res[i]['cumret'], back_trade_res[i]['cumret']
                
            res = {
                'area' : agg_area,
                'df_premium' : agg_premium,
                'df_ret' : agg_ret,
                'df_cumret' : agg_cum_ret,
                'daily_ret' : agg_daily_ret,
                'cumret' : agg_cumret
                }
            res_list.append(res)
        
        return res_list