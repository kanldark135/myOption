#%% 
import pandas as pd
import numpy as np
import option_calc as calc
import compute
from datetime import datetime
import sql

#%% 
# #1. 데이터 불러오기 sqlite db

# local_file_path = './option_k200.db'

# conn = sql.db_connect(local_file_path)

# query_select = '''
#                 SELECT * FROM MONTHLY_TOTAL
# '''
# # query_select = '''
# #                 SELECT * FROM weekly_total
# # '''

# df_raw = pd.read_sql(query_select, conn)
# conn.close()


#%% 데이터 불러오기 pkl

monthly = pd.read_pickle("./data_pickle/df_monthly.pkl")
call_df_raw = monthly.loc[monthly['cp'] == 'C']
put_df_raw = monthly.loc[monthly['cp'] == 'P']

#%% 
# number_of_contracts vol-based dynamic sizing 구현
# 콜이랑 풋이랑 동시에 할수 있게 > 스트랭글 등등
# 복리로 투자했으면 어떻게 됬을지? 누적수익률 구하는 함수

#%% general 함수 : option function 으로 옮길것

# raw data 에서 옵션 pivot_table 구하는 함수

def get_pivot_chain_within_group(raw_df, values = ['adj_price', 'iv_interp', 'delta']):

    # multiindex dataframe = [델타-행사가들 / 가격-행사가들, 등가, dte, 종가...] 생성

    res = pd.pivot_table(raw_df, values = values, index = raw_df.index, columns = ['strike'], aggfunc='last')
    aux = pd.pivot_table(raw_df, values = ['atm', 'dte', 'close'], index = raw_df.index)
    aux.columns = pd.MultiIndex.from_tuples([(col, "") for col in aux.columns])
    res = pd.concat([res, aux], axis = 1, join = 'inner')

    return res

# myentrydate / exitdate : 온갖 방법으로 진입 /청산시점 구할 것이기 때문에 어떤 함수로 방법론을 특정할 수가 없음
# 어디선가 진입 /청산 df 구해와서 그 datetimeIndex 만 가져온다고 치고 거기서부터 시작하기

#1. 스토캐스틱 과열 기준 진입일
import get_entry_date
k200 = pd.read_pickle("./data_pickle/k200.pkl")
ta_based_entry = get_entry_date.contrarian(k200)
stoch = ta_based_entry.stoch_rebound(k = 5, d = 3, smooth_d = 3)
entry_dates_stoch = stoch.loc[stoch['signal_stoch'] == 1].index

# 2. 월요일마다 진입 (월 = 0 ~ 일 = 6)
entry_dates_monday = (monthly.index.unique()[monthly.index.unique().weekday == 0])

trade_spec = 

def create_trade_entries_one_expiry(df,
                    entry_dates,
                    dist_from_atm : list,
                    number_of_contracts : list,
                    dte_range = [35, 70]
                   ):
    '''
    return trade_dict with keys
    entry_date : one entry date in datetime format
    strikes : list of multiple ks
    number_of_contracts : trade size of each options in the same order as k
    '''
    def get_entry_dates(df, entry_dates):
        res = df.loc[df.index.isin(entry_dates)].index
        return res
    
    entry_dates = get_entry_dates(df = df, entry_dates = entry_dates)

    # 1) 해당 subgroup 내에서 포지션 잡는 로우만 식별
    res = df.loc[entry_dates]

    # 2) 특정 만기 이내 dte만 사용하여 진입
    res = res.loc[res['dte'].isin(range(dte_range[0], dte_range[1]))]

    # 3) 전략 구현 (= 각 옵션들 행사가 선택))
    
    '''
    # 'number' : dist_from_atm 을 atm 대비 벌어진 값으로 설정되게
    # 'pct' : dist_from_atm 을 atm 대비 벌어진 수준(%) 구해서 알아서 설정되게
    # 'delta' : dist_from_atm 을 해당 시점 델타 기준으로 알아서 설정되게 (델타 20/15 -> 행사가 알아서 선정)

    ---> 위에 세개 전부 혼용 가능하도록 (진입시 "0.04%에 긋고 / +7.5 행사 위에다가 매도 후 / 델타 0.05짜리로 외가헤지" 와 같은 전략 구현)
    '''
    def find_strikes(row, dist_from_atm):

        '''
        row 는 더미값으로 사실상 df.apply(axis = 1) 의 각 행을 변수로 그대로 받는 목적으로 선언
        '''
        row = row.astype('float64')
        res = []
        for key, value in dist_from_atm:
            if key == 'number':
                strike  = row['atm'].squeeze() + value
            elif key == 'pct':
                raw_value = row['close'].squeeze() * (1 + value) 
                strike = calc.get_closest_strike(raw_value)
            elif key == 'delta':
                strike = np.abs((np.abs(row['delta']) - np.abs(value))).astype('float64').idxmin()        
            res.append(strike)

        return res
    
    res['trade_dict'] = res.apply(lambda row : {
        'entry_date' : row.name,
        'strikes' : find_strikes(row, dist_from_atm = dist_from_atm),
        'number_of_contracts' : number_of_contracts
        }, axis = 1)

    # 4) dynamic sizing
    return res['trade_dict'].tolist()

def get_trade_result(df, trade_dict: dict, is_complex_strat = False, profit_take = 0.5, stop_loss = 2):

    df = df['adj_price'] # dataframe 전체에서 가격 부분만 사용

    ''' trade_dict
    entry_date : datetime
    strikes : list. multiple ks
    number_of_contracts : list. how many options to trade?
    
    complex_strat = True 인 경우 (BWB 와 같이 목표손익이 initial credt/debit과 관계 없는 경우)
    profit / loss 값 = 목표손익 포인트

    complex_strat = False 인 경우
    profit / loss 값 = inital credit / debit 의 배수
    '''

    df_trade_area = df.loc[trade_dict['entry_date'] : , trade_dict['strikes']]
    df_pos_net_premium = df_trade_area.multiply(np.negative(trade_dict['number_of_contracts']), axis = 1)
    df_profit = (df_pos_net_premium.shift(1) - df_pos_net_premium).fillna(0)
    df_cum_profit = df_profit.cumsum()
    daily_profit = df_profit.sum(axis = 1)
    cum_profit = df_cum_profit.sum(axis = 1)

    # 익손절 구현

    if is_complex_strat == True:
        
        profit_target = profit_take
        loss_target = stop_loss

    else:
        profit_target = np.abs(df_pos_net_premium.iloc[0].sum().squeeze()) * profit_take
        loss_target = np.abs(df_pos_net_premium.iloc[0].sum().squeeze()) * stop_loss
        
    # 익절/손절 index 식별 = 익절 또는 손절 나갔을 모든상황중 가장 처음 index 에 해당하는 날짜

    # IndexError 발생상황 1 : 중간청산이 안 되는경우 (익절 또는 손절 안되고 그대로 만기까지 가는 경우) : liquidate_date = 만기로 설정
    try:
        liquidate_date = cum_profit[
            (cum_profit >= profit_target)|
            (cum_profit <= loss_target)
        ].index[0]

    except IndexError:
        liquidate_date = None

    df_trade_area = df_trade_area.loc[:liquidate_date]
    df_pos_net_premium = df_pos_net_premium.loc[:liquidate_date] 
    df_profit = df_profit.loc[:liquidate_date]
    df_cum_profit = df_cum_profit.loc[:liquidate_date]
    daily_profit = daily_profit.loc[:liquidate_date]
    cum_profit = cum_profit.loc[:liquidate_date]

    res = {
        'area' : df_trade_area,
        'df_premium' : df_pos_net_premium,
        'df_profit' : df_profit,
        'df_cum_profit' : df_cum_profit,
        'daily_ret' : daily_profit,
        'cum_ret' : cum_profit
    }

    return res

def get_agg_return(df, dist_from_atm, number_of_contracts, preferred_weekday = 0, dte_range = [35,70], is_complex_strat = False, profit_take = 0.5, stop_loss = 2):

    res_list = []

    trade_list = create_trade_entries_one_expiry(df, dist_from_atm = dist_from_atm, number_of_contracts = number_of_contracts, preferred_weekday = preferred_weekday, dte_range = dte_range)

    for trade in trade_list:
        trade_res = get_trade_result(df, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['daily_ret']
        res_list.append(trade_res)

    daily_ret = pd.concat(res_list, ignore_index = True, axis = 1).sum(axis = 1)

    res = {
        'res_list' : res_list,
        'daily_ret' : daily_ret
    }

    return res

def get_final_result(df_raw_subgroup, 
                     dist_from_atm, 
                     number_of_contracts, 
                     preferred_weekday = 0, 
                     dte_range = [35,70], 
                     is_complex_strat = False, 
                     profit_take = 0.5, 
                     stop_loss = 2):
    
    ''' trade_dict
    entry_date : datetime
    strikes : list. multiple ks
    number_of_contracts : list. how many options to trade?
    
    complex_strat = True 인 경우 (BWB 와 같이 목표손익이 initial credt/debit과 관계 없는 경우)
    profit / loss 값 = 목표손익 포인트

    complex_strat = False 인 경우
    profit / loss 값 = inital credit / debit 의 배수
    '''

    ''' 내가 저장해놓은 melt 된 데이터형태에서 필요한 값으로만 pivot 구성하기'''
    try:
        df = df_raw_subgroup.pipe(get_pivot_chain_within_group, values = ['adj_price', 'iv_interp', 'delta'])
        res = get_agg_return(df, dist_from_atm = dist_from_atm, number_of_contracts = number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)
        res = res['daily_ret']
    except:
        res = None

    return res

#%%

if __name__ == "__main__":

    data_from = '2010-01-15' # 옛날에는 행사가가 별로 없어서 전략이 이상하게 나감
    
    # 70-35일 남은 차월물 / 월요일마다 / 델타 0.1 / 양매도 / 중간 익손절 적용 (50% / -200%)

    put_dist_from_atm = [
        ('delta', -0.25),
        ('delta', -0.22),
        ('delta', -0.05),
    ]
    put_number_of_contracts = [1, -1, -1]

    call_dist_from_atm = [
        ('delta', 0.25),
        ('delta', 0.17),
        ('delta', 0.08)
    ]
    call_number_of_contracts = [5, -12, 5]
    
    preferred_weekday = 0
    dte_range = [35, 70]

    is_complex_strat = True
    profit_take = 4
    stop_loss = -4

    grouped_call = call_df_raw.loc[data_from:].groupby('expiry')
    grouped_put = put_df_raw[data_from:].groupby('expiry')

    all_expiry = grouped_put.groups.keys()

    # 얼토당토않은 가격 조정하는 함수 : 구해놓고 나중에 아예 DB에 쳐박아놓을것

# 테스트용 예시 : 2008-02-14 만기따리

    sample = grouped_call.get_group('2010-04-08')
    cdf = sample.pipe(get_pivot_chain_within_group)
    ctrade_list = create_trade_entries_one_expiry(cdf, call_dist_from_atm, call_number_of_contracts)
    cret = get_agg_return(cdf, call_dist_from_atm, call_number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)

    cres_n = []

    for trade in ctrade_list:
        cres = pd.concat([get_trade_result(cdf, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_trade_result(cdf, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')
        cres_n.append(cres)

    cres = get_final_result(sample, call_dist_from_atm, call_number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)

    sample_2 = grouped_put.get_group('2010-04-08')
    pdf = sample_2.pipe(get_pivot_chain_within_group)
    ptrade_list = create_trade_entries_one_expiry(pdf, put_dist_from_atm, put_number_of_contracts)
    pret = get_agg_return(pdf, put_dist_from_atm, put_number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)

    pres_n = []

    for trade in ptrade_list:
        pres = pd.concat([get_trade_result(pdf, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_trade_result(pdf, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')
        pres_n.append(pres)

    pres = get_final_result(sample_2, put_dist_from_atm, put_number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)

#%%  실전 분석

result_put = grouped_put.apply(get_final_result, 
                dist_from_atm = put_dist_from_atm, 
                number_of_contracts = put_number_of_contracts, 
                preferred_weekday = preferred_weekday, 
                dte_range = dte_range,
                is_complex_strat = is_complex_strat, 
                profit_take = profit_take, 
                stop_loss = stop_loss)

#%% 

result_call = grouped_call.apply(get_final_result, 
                dist_from_atm = call_dist_from_atm, 
                number_of_contracts = call_number_of_contracts, 
                preferred_weekday = preferred_weekday, 
                dte_range = dte_range,
                is_complex_strat = is_complex_strat, 
                profit_take = profit_take, 
                stop_loss = stop_loss)

