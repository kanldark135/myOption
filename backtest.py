#%% 
import pandas as pd
import numpy as np
import option_calc as calc
import compute
from datetime import datetime

#%% 
# #1. 데이터 불러오기 sqlite db

# local_file_path = 'C:/Users/kanld/Desktop/option_k200.db'

# conn = sql.db_connect(local_file_path)

# query_select = '''
#                 SELECT * FROM MONTHLY_TOTAL
# '''
# # query_select = '''
# #                 SELECT * FROM weekly_total
# # '''

# df_raw = pd.read_sql(query_select, conn)
# conn.close()


#%%
# import get_entry_date

# df = pd.read_pickle('./working_data/df_monthly.pkl')
# df_k200 = pd.read_pickle('./working_data/df_k200.pkl')

# grouped = df.groupby('expiry')
# all_expiry = grouped.groups.keys()

# trade_spec = {'P' : [('delta', -0.2, -1)]}
# dte_range = [42, 70]
# is_complex_strat = False
# profit_take = 0.5
# stop_loss = -2

# entry_cond1 = get_entry_date.weekday_entry(df_k200, [0, 4])
# entry_cond2 = df_k200.trend.psar_trend('l')
# entry_dates = get_entry_date.get_date_intersect(df, entry_cond1, entry_cond2)

# exit_cond1 = df_k200.contra.psar_rebound('s')
# exit_dates = get_entry_date.get_date_intersect(df, exit_cond1)

# sample = grouped.get_group('2020-03-12')
# sample_pivoted = sample.pipe(get_pivot_table)

# #1.
# trades = create_trade_entries(sample_pivoted, entry_dates, trade_spec)

# #2.
# res = get_single_trade_result(sample_pivoted, trades[1])

# trade_res = list(map(lambda trade : get_single_trade_result(sample_pivoted, trade), trade_entry))
# trade_res_stopped = list(map(lambda result : stop_single_trade(result, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss), trade_res))

# ret = get_single_expiry_result(df_pivoted = sample_pivoted, 
#                     entry_dates = entry_dates, 
#                     trade_spec = trade_spec,
#                     dte_range = dte_range,                       
#                     is_complex_strat = is_complex_strat, 
#                     profit_take = profit_take, 
#                     stop_loss = stop_loss)

# result = get_vertical_trade_result(sample,
#                     entry_dates = entry_dates,
#                     trade_spec = trade_spec,
#                     dte_range = dte_range,                        
#                     is_complex_strat = is_complex_strat, 
#                     profit_take = profit_take, 
#                     stop_loss = stop_loss
#                     )


#%% general 함수 : option function 으로 옮길것

# raw data 에서 옵션 pivot_table 구하는 함수

def get_pivot_table(raw_df, values = ['adj_price', 'iv_interp', 'delta']):

    call = raw_df[raw_df['cp'] == "C"]
    put = raw_df[raw_df['cp'] == "P"]

    # multiindex dataframe = [델타-행사가들 / 가격-행사가들, 등가, dte, 종가...] 생성
    def create_pivot_table(df):
        res = pd.pivot_table(df, values = values, index = df.index, columns = ['strike'], aggfunc='last')
        res = res.rename(columns = {"adj_price" : "price", 'iv_interp' : "iv"}, level = 0)
        return res
    
    def create_new_multicolumn(df, cp : str):
        level_1 = [cp] * len(df.columns)
        level_2 = list(zip(*df.columns))[0]
        level_3 = list(zip(*df.columns))[1]
        new_column = pd.MultiIndex.from_arrays([level_1, level_2, level_3])
        df.columns = new_column
        return df

    call_pivot = call.pipe(create_pivot_table).pipe(create_new_multicolumn, cp = "C")
    put_pivot = put.pipe(create_pivot_table).pipe(create_new_multicolumn, cp = "P")

    res = pd.concat([call_pivot, put_pivot], axis = 1)

    aux = pd.pivot_table(raw_df, values = ['atm', 'dte', 'close', 'vkospi'], index = raw_df.index)
    aux.columns = pd.MultiIndex.from_tuples([(col, "", "") for col in aux.columns])
    res = pd.concat([res, aux], axis = 1, join = 'inner')

    return res

def create_trade_entries(df_pivoted,
                    entry_dates,
                    trade_spec,
                    dte_range = [35, 70]
                   ):
    '''
    trade_spec = {"C" : [('number', 20, 5), ('pct', 0.06, -10)], "P" : [('delta', -0.10, -3)]} 의 꼴로
    각각 콜 풋 양쪽에서 묶음으로 트레이드하는 경우 일거에 기입
    '''
    def get_entry_dates(df, entry_dates):
        res = df.loc[df.index.isin(entry_dates)].index
        return res
    
    entry_dates = get_entry_dates(df = df_pivoted, entry_dates = entry_dates)

    # 1) 해당 subgroup 내에서 포지션 잡는 로우만 식별
    df = df_pivoted.loc[entry_dates]

    # 2) 특정 만기 이내 dte만 사용하여 진입
    df = df.loc[df['dte'].isin(range(dte_range[0], dte_range[1]))]

    # 3) 전략 구현 (= 각 옵션들 행사가 선택))

    '''
    # 'number' : dist_from_atm 을 atm 대비 벌어진 값으로 설정되게
    # 'pct' : dist_from_atm 을 atm 대비 벌어진 수준(%) 구해서 알아서 설정되게
    # 'delta' : dist_from_atm 을 해당 시점 델타 기준으로 알아서 설정되게 (델타 20/15 -> 행사가 알아서 선정)
    '''
    def create_trade(row, trade_spec):
  
        '''
        row 는 더미값으로 사실상 df.apply(axis = 1) 의 각 행을 변수로 그대로 받는 목적으로 선언
        '''
        row = row.astype('float64')
        trade = []
        contracts = []
        for key in trade_spec.keys():
            for single_leg in trade_spec.get(key):
                if single_leg[0] == 'number': # 자체 계산
                    strike  = row['atm'].squeeze() + single_leg[1]
                elif single_leg[0] == 'pct': # 자체 계산
                    raw_value = row['close'].squeeze() * (1 + single_leg[1]) 
                    strike = calc.get_closest_strike(raw_value)
                elif single_leg[0] == 'delta': # 테이블에서 lookup
                    strike = np.abs((np.abs(row.loc[(key, 'delta')]) - np.abs(single_leg[1]))).astype('float64').idxmin()
                
                idx = (key, "price", strike)
                size = single_leg[2]
                trade.append(idx)
                contracts.append(size)

        return [trade, contracts]
    
    res = df.apply(lambda row : {
        'entry_date' : row.name,
        'trade' : create_trade(row, trade_spec = trade_spec)[0],
        'contract' : create_trade(row, trade_spec = trade_spec)[1]}, axis = 1
        )
    try:
        res = res.tolist()
    except AttributeError: # 해당 만기에 거래 1도 없는경우 dummy trade 생성해서 밑에서 0 trade로 환원
        res = {'entry_date' : '9999-01-01', 'trade' : [], 'contract' : 0}

    return res

def get_single_trade_result(df_pivoted, single_trade: dict):

    ''' trade_dict
    entry_date : datetime
    trade : list
    contract : list
    '''
    try:
        entry_date = single_trade['entry_date']
        df_trade_area = df_pivoted.loc[entry_date : , single_trade['trade']]
        df_net_premium = df_trade_area.multiply(np.negative(single_trade['contract']), axis = 1)
        df_ret = - df_net_premium.diff(1).fillna(0)
        daily_ret = df_ret.sum(axis = 1)
        cumret = df_ret.cumsum().sum(axis = 1)

    # 트레이드가 아예 없는 경우 발생하는 에러 처리 (IndexError TypeError)
    # 옛날데이터의 경우 행사가가 없어서 산정해놓은 행사가에 매칭되는 자료가 없는 경우 (Keyerror) -> 그냥 안한 셈 치기

    except (IndexError, TypeError, KeyError):
        df_trade_area = pd.Series(0, index = df_pivoted.index)
        df_net_premium = pd.Series(0, index = df_pivoted.index)        
        df_ret = pd.Series(0, index = df_pivoted.index)
        daily_ret = pd.Series(0, index = df_pivoted.index)
        cumret = pd.Series(0, index = df_pivoted.index)

    res = {
        'area' : df_trade_area,
        'df_premium' : df_net_premium,
        'df_ret' : df_ret,
        'daily_ret' : daily_ret,
        'cumret' : cumret
    }

    return res

def stop_single_trade(trade_result : dict,
                    exit_dates = [],
                    stop_dte = 0,
                    is_complex_strat = False,
                    profit_take = 0.5,
                    stop_loss = -2):
    '''
    complex_strat = True 인 경우 (BWB 와 같이 목표손익이 initial credt/debit과 관계 없는 경우)
    profit / loss 값 = 목표손익 포인트

    complex_strat = False 인 경우
    profit / loss 값 = inital credit / debit 의 배수
    '''
    try:
        initial_premium = trade_result['df_premium'].iloc[0].sum().squeeze()
    except AttributeError:
        initial_premium = trade_result['df_premium'].iloc[0].sum()
    
    cumret = trade_result['cumret']
    try:
        dummy = trade_result['cumret'].index[-1] - pd.DateOffset(days = stop_dte)
        dte_stop = trade_result['cumret'].index[dummy <= trade_result['cumret'].index][0] 
        # 논리는 stop_dte 보다 작은 날중 가장 큰 dte날 중간에 휴일있으면 "휴일 지내고" 청산 처리
    except IndexError:
        dte_stop = pd.Timestamp('2099-01-01')

    # IndexError 발생상황 : 해당 월물 기간동안 Hard Stop 없는경우 -> 없는걸로 처리
    try:
        hard_stop = cumret.loc[cumret.index.isin(exit_dates)].index[0]
    except IndexError:
        hard_stop = pd.Timestamp('2099-01-01')

    profit_target = profit_take if is_complex_strat else max(np.abs(initial_premium) * profit_take, 0.01)
    loss_target = stop_loss if is_complex_strat else min(np.abs(initial_premium) * stop_loss, -0.01)
        
    # IndexError 발생상황 : 중간청산이 안 되는경우 (익손절 안되고 만기까지 가는 경우)

    try:
        stop_date = cumret[
            (cumret >= profit_target)|
            (cumret <= loss_target)
        ].index[0]
    except IndexError:
        stop_date = pd.Timestamp('2099-01-01')

    # 최종적으로 profit/loss 기준 익손절과 시그널상 custom exit date 간에 더 빠른 날짜 적용
    
    liquidate_date = np.nanmin([dte_stop, hard_stop, stop_date]) 

    df_trade_area = trade_result['area'].loc[:liquidate_date]
    df_net_premium = trade_result['df_premium'].loc[:liquidate_date] 
    df_ret = trade_result['df_ret'].loc[:liquidate_date]
    daily_ret = trade_result['daily_ret'].loc[:liquidate_date]
    cumret = trade_result['cumret'].loc[:liquidate_date]

    res = {
    'area' : df_trade_area,
    'df_premium' : df_net_premium,
    'df_ret' : df_ret,
    'daily_ret' : daily_ret,
    'cumret' : cumret
    }

    return res

def get_single_expiry_result(df_pivoted,
                         entry_dates,                        
                         trade_spec,
                         dte_range = [35,70],
                         exit_dates = [],
                         stop_dte = 0,                     
                         is_complex_strat = False,
                         profit_take = 0.5,
                         stop_loss = -2):

    '''한 만기 내에서 모든 진입시점 만들고 / 각 진입에 대한 만기까지의 손익 및 / 중간익손절까지 반영하여 => 
    각 매매의 결과 (=result_list) list / 전부 합산한 해당 만기의 일일손익 output'''

    #1. 매매 엔트리 생성
    trade_entries = create_trade_entries(df_pivoted, entry_dates, trade_spec, dte_range)
    #2. 엔트리별로 만기까지 보유시의 df 생성
    trade_res = list(map(lambda trade : get_single_trade_result(df_pivoted, trade), trade_entries))
    #3. 만기보유 df에 stop 조건 적용
    trade_res_stopped = list(map(lambda idx : stop_single_trade(trade_res[idx],
                                                        exit_dates = exit_dates,
                                                        stop_dte = stop_dte,
                                                        is_complex_strat = is_complex_strat,
                                                        profit_take = profit_take,
                                                        stop_loss = stop_loss).get('daily_ret'), range(len(trade_res))))

    trade_summary_stopped = list(map(lambda idx : stop_single_trade(trade_res[idx],
                                                        exit_dates = exit_dates,
                                                        is_complex_strat = is_complex_strat,
                                                        stop_dte = stop_dte,
                                                        profit_take = profit_take,
                                                        stop_loss = stop_loss).get('df_ret'), range(len(trade_res))))

#4. 결과 합산해서 해당 expiry 전체 일수익률 도출
    try: 
        daily_ret = pd.concat(trade_res_stopped, axis = 1).sum(axis = 1)
    except ValueError:
        daily_ret = pd.Series(0, index = df_pivoted.index)

    all_trades = {}
    number_of_trades = 0
    number_of_winners = 0

    for idx in range(len(trade_summary_stopped)):
        single_trade_res = trade_summary_stopped[idx].dropna()
        try:
            single_trade_res_summed = single_trade_res.sum(axis = 1, skipna = False)
            final_ret = single_trade_res_summed.cumsum().iloc[-1].squeeze()

            all_trades[pd.to_datetime(trade_entries[idx].get('entry_date'))] = {
            'trade_ret' : single_trade_res, # 해당 만기 내 개별 trade 들의 각각의 일수익금 df
            'final_ret' : final_ret, # 해당 만기 내 개별 trade 들의 각각의 최종수익금
            'trade_drawdown' : single_trade_res_summed.min(),
            # 해당 만기 내 개별 trade 의 운용 중 "당일 하루" 최대 평가손실 (손절쳤으면 확정손실). trade_ret 상의 누적손실이 금액으로는 더 클수 있음
            'drawdown_date' : single_trade_res_summed.astype('float64').idxmin()
            # drawdown 발생한 날 / daily_ret 에서 식별하는 날과 다름 (이건 entry_date)
            }
            number_of_trades = number_of_trades + 1 # 매매 누적수익률이 0이라는건 해당 월물에 매매 한건도 없다는거임
            if final_ret > 0:
                number_of_winners = number_of_winners + 1

        except ValueError:
            pass

    all_trades = pd.DataFrame(all_trades).T
    summary = {'n' : number_of_trades, 'win' : number_of_winners}
    
    res = {'daily_ret' : daily_ret, 'all_trades' : all_trades, 'summary' : summary}

    return res

def get_vertical_trade_result(df, 
                     entry_dates, 
                     trade_spec,
                     dte_range = [35,70], 
                     exit_dates = [],
                     stop_dte = 0,
                     is_complex_strat = False, 
                     profit_take = 0.5, 
                     stop_loss = -2):
    
    grouped = df.groupby('expiry')
    all_expiry = list(grouped.groups.keys())

    daily_ret_list = {}
    all_trades = []
    number_of_trades = 0
    number_of_winners = 0

    for expiry in all_expiry:
        df = grouped.get_group(expiry)
        df = df.pipe(get_pivot_table, values = ['adj_price', 'iv_interp', 'delta'])
        res = get_single_expiry_result(df, entry_dates = entry_dates,
                                   trade_spec = trade_spec,
                                   dte_range = dte_range,
                                   exit_dates = exit_dates,
                                   stop_dte = stop_dte,
                                   is_complex_strat = is_complex_strat,
                                   profit_take = profit_take,
                                   stop_loss = stop_loss)
        daily_ret_list[expiry] = res['daily_ret']
        
        all_trades.append(res['all_trades'])
        
        single_summary = res['summary']
        number_of_trades += single_summary.pop('n')
        number_of_winners += single_summary.pop('win')

    # 계좌 총 일일 손익
    daily_ret = pd.DataFrame(daily_ret_list).stack().swaplevel(0, 1)
    # 진행했던 모든 매매들 및 각 누적수익. entry_date 을 index로 dataframe 화 (dataframe inside dataframe)
    all_trades = pd.concat(all_trades, axis = 0)

    # 주요 통계
    total_ret = all_trades['final_ret'].sum()
    avg_ret = all_trades['final_ret'].mean() # 실현수익 기준 평균 금액 수익
    avg_win = all_trades['final_ret'].loc[all_trades['final_ret'] > 0].mean()
    avg_loss = all_trades['final_ret'].loc[all_trades['final_ret'] < 0].mean()
    account_volatility = daily_ret.std() # 계좌총액 기준 금액 변동성
    strat_mdd = (all_trades['trade_drawdown'].min(), all_trades['drawdown_date'].loc[all_trades['trade_drawdown'].astype('float64').idxmin()])
    net_liq_mdd = (daily_ret.min(), daily_ret.idxmin()[1])
    
    summary = {'n' : number_of_trades,
    'win' : number_of_winners,
    'total_ret' : total_ret,
    'avg_ret' : avg_ret,
    'avg_win' : avg_win,
    'avg_loss' : avg_loss,
    'vol' : account_volatility,
    'single_strat_mdd' : strat_mdd,
    'net_liq_mdd' : net_liq_mdd
    }
    
    final_res = {'daily_ret' : daily_ret,
    'all_trades' : all_trades,
    'summary' : summary
    }

    return final_res

## get calendar trade_result + list
def get_calendar_trade_result(df_monthly, 
                              entry_dates, 
                              front_spec, 
                              back_spec,
                              front_dte = [14, 35], 
                              back_dte = [28, 77],
                              exit_dates = [],
                              stop_dte = 0,
                              is_complex_strat = False,
                              profit_take = 2,
                              stop_loss = -2):
    
    # raw 데이터 바로 feed
    grouped = df_monthly.groupby('expiry')
    all_expiry = grouped.groups.keys()

    # 근월 / 차월 만기 pair 생성하는 함수
    def get_pair_expiry(all_expiry):
        all_expiry = list(all_expiry)
        res = list(zip(all_expiry, all_expiry[1:]))
        return res

    # 근월물 진입시점에 맞춰서 차월물 진입시점 일치시키는 함수
    def get_filtered_trades_back(trade_front, trade_back):
        front_dates = list(map(lambda trade : trade['entry_date'], trade_front))
        filtered_trades = list(filter(lambda trade : trade['entry_date'] in front_dates, trade_back))
        return filtered_trades

    # (근월만기, 차월만기) 리스트 도출
    paired_expiry = get_pair_expiry(all_expiry)
    
    daily_ret_list = {}
    all_trades = {}
    number_of_trades = 0
    number_of_winners = 0

    for front, back in paired_expiry: # 모든 (근월만기, 차월만기) 리스트에 대해서
    
    # 1. 각각 근월물 / 차월물 데이터프레임 도출
        df_front = grouped.get_group(front)
        df_back = grouped.get_group(back)
    # 2. 각각 피봇화
        front_pivoted = get_pivot_table(df_front)
        back_pivoted = get_pivot_table(df_back)
    # 3. 각각 피봇에서 해당 월물에서 도출되는 모든 trades 도출 -> 이때 차월물 trade는 근월물 trade랑 진입시점 일치
        front_trade_entry = create_trade_entries(front_pivoted, entry_dates = entry_dates, trade_spec = front_spec, dte_range = front_dte)
        back_trade_entry = create_trade_entries(back_pivoted, entry_dates = entry_dates, trade_spec = back_spec, dte_range = back_dte)
        filtered_back_trade_entry = get_filtered_trades_back(front_trade_entry, back_trade_entry)
    # 4. 각 진입 trades 들에 대한 실제 만기까지 손익 df 도출
        front_trade_res = list(map(lambda trade : get_single_trade_result(front_pivoted, trade), front_trade_entry))
        back_trade_res = list(map(lambda trade : get_single_trade_result(back_pivoted, trade), filtered_back_trade_entry))
    # 5. 근월 df/차월df 합산하는 함수 정의 및 실시
        def result_aggregate(front_trade_res, back_trade_res):
            res_list = []

            try:
                for i in range(len(front_trade_res)):
                    agg_area = pd.concat([front_trade_res[i]['area'], back_trade_res[i]['area']], axis = 1)
                    agg_premium = pd.concat([front_trade_res[i]['df_premium'], back_trade_res[i]['df_premium']], axis = 1)
                    agg_ret = pd.concat([front_trade_res[i]['df_ret'], back_trade_res[i]['df_ret']], axis = 1)
                    agg_daily_ret = front_trade_res[i]['daily_ret'] + back_trade_res[i]['daily_ret']
                    agg_cumret = front_trade_res[i]['cumret'] + back_trade_res[i]['cumret']
                        
                    res = {
                        'area' : agg_area,
                        'df_premium' : agg_premium,
                        'df_ret' : agg_ret,
                        'daily_ret' : agg_daily_ret,
                        'cumret' : agg_cumret
                        }
                    res_list.append(res)

            except IndexError:
                agg_area = pd.Series(0, index = df_front.index)
                agg_premium = pd.Series(0, index = df_front.index)        
                agg_ret = pd.Series(0, index = df_front.index)
                agg_daily_ret = pd.Series(0, index = df_front.index)
                agg_cumret = pd.Series(0, index = df_front.index)
            
            return res_list
        
        agg_trade_res = result_aggregate(front_trade_res, back_trade_res)

    # 6. 합산손익에 대한 stop 조건 적용
        trade_res_stopped = list(map(lambda idx : stop_single_trade(agg_trade_res[idx],
                                                                    exit_dates = exit_dates,
                                                                    stop_dte = stop_dte,
                                                                    is_complex_strat = is_complex_strat, 
                                                                    stop_loss = stop_loss, 
                                                                    profit_take = profit_take).get('daily_ret'),
                                                                    range(len(agg_trade_res))))

        trade_summary_stopped = list(map(lambda idx : stop_single_trade(agg_trade_res[idx],
                                                            exit_dates = exit_dates,
                                                            stop_dte = stop_dte,
                                                            is_complex_strat = is_complex_strat,
                                                            profit_take = profit_take,
                                                            stop_loss = stop_loss).get('df_ret'), 
                                                            range(len(agg_trade_res))))
        
    # 7. stop 조건까지 적용된 특정 (근월만기, 차월만기) 에 대한 최종 합산 daily return 도출
        try:
            daily_ret = pd.concat(trade_res_stopped, axis = 1).sum(axis = 1)
        except ValueError:
            daily_ret = pd.Series(0, index = front_pivoted.index)
        
        daily_ret_list[front] = daily_ret
    
    # 8. loop 내에서 통계용 summary 도 같이 도출

        for idx in range(len(trade_summary_stopped)):
            single_trade_res = trade_summary_stopped[idx].dropna()
            try:
                single_trade_res_summed = single_trade_res.sum(axis = 1, skipna = False)
                final_ret = single_trade_res_summed.cumsum().iloc[-1].squeeze()

                # 캘린더는 근월물 + 차월물 합산할때 근월물의 trade_res 에서는 만기 이후 값은 NA
                # 차월물의 trade_res 에서는 여전히 값 남아 있음
                # 이때 sum(axis = 1) 에서 skipna = True 그대로 두면 차월물 평가액이 포함되게 됨
                # 원칙적으로 근월물 만기에 차월물도 같이 청산한다는 가정이므로 근월물 만기 이후의 값은 다 NA 처리 + 삭제

                all_trades[pd.to_datetime(front_trade_entry[idx].get('entry_date'))] = {
                'trade_ret' : single_trade_res, # 해당 만기 내 개별 trade 들의 각각의 일수익금 df
                'final_ret' : final_ret , # 해당 만기 내 개별 trade 들의 각각의 최종수익금
                'trade_drawdown' : single_trade_res_summed.min(),
                # 해당 만기 내 개별 trade 의 운용 중 최대 평가손실 (손절쳤으면 확정손실)
                'drawdown_date' : single_trade_res_summed.astype('float64').idxmin()
                # drawdown 발생한 날 / daily_ret 에서 식별하는 날과 다름 (이건 entry_date)
                }                
                
                number_of_trades = number_of_trades + 1 # 매매 누적수익률이 0이라는건 해당 월물에 매매 한건도 없다는거임
                if final_ret > 0:
                    number_of_winners = number_of_winners + 1

            except ValueError:
                pass
        
    # 8. 위에 모든 (근월, 차월) 에 대해서 loop 돌린 daily_ret_list 를 단일만기 trade랑 동일한 구조로 변경
    daily_ret = pd.DataFrame(daily_ret_list).T.stack()

    # 9. 기타 summary 정리
    all_trades = pd.DataFrame(all_trades).T

    total_ret = all_trades['final_ret'].sum()
    avg_ret = all_trades['final_ret'].mean() # 실현수익 기준 평균 금액 수익
    avg_win = all_trades['final_ret'].loc[all_trades['final_ret'] > 0].mean()
    avg_loss = all_trades['final_ret'].loc[all_trades['final_ret'] < 0].mean()
    account_volatility = daily_ret.std() # 계좌총액 기준 금액 변동성
    strat_mdd = (all_trades['trade_drawdown'].min(), all_trades['drawdown_date'].loc[all_trades['trade_drawdown'].astype('float64').idxmin()])
    net_liq_mdd = (daily_ret.min(), daily_ret.idxmin()[1])

    summary = {'n' : number_of_trades,
    'win' : number_of_winners,
    'total_ret' : total_ret,
    'avg_ret' : avg_ret,
    'avg_win' : avg_win,
    'avg_loss' : avg_loss,
    'vol' : account_volatility,
    'single_strat_mdd' : strat_mdd,
    'net_liq_mdd' : net_liq_mdd
    }
    
    final_res = {'daily_ret' : daily_ret,
    'all_trades' : all_trades,
    'summary' : summary
    }
    
    return final_res
