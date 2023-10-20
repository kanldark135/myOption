#%% 
import pandas as pd
import numpy as np
import option_calc as calc
import compute
from datetime import datetime


#%% 
#1. 데이터 불러오기

call_df_raw = pd.read_pickle("./data_pickle/call_monthly.pkl")
put_df_raw = pd.read_pickle("./data_pickle/put_monthly.pkl")

cycle = 'back'

cb_price = calc.get_option_data('price', cycle = 'back', callput = 'call', moneyness_lb = -30, moneyness_ub = 50)
cb_iv = calc.get_option_data('iv', cycle = 'back', callput = 'call', moneyness_lb = -30, moneyness_ub = 50)

pb_price = calc.get_option_data('price', cycle = 'back', callput = 'put', moneyness_lb = -30, moneyness_ub = 50)
pb_iv = calc.get_option_data('iv', cycle = 'back', callput = 'put', moneyness_lb = -30, moneyness_ub = 50)


# 내가옵션 일정 수준 이상 내가격 가면 거래안되서 0원으로 가격 비는거 처리
# 1) 일일히 bsm 으로 계산하기 -> 볼이 없어서 정확한 계산 안 됨 / interpolation 가능하나 굳이...?
# -> 2) 그냥 내재가치로 퉁 치기 VVV
# 3) 아예 냅두기

# number_of_contracts vol-based dynamic sizing 구현
# 콜이랑 풋이랑 동시에 할수 있게 > 스트랭글 등등
# 복리로 투자했으면 어떻게 됬을지? 누적수익률 구하는 함수

#%% general 함수 : option function 으로 옮길것

# raw data 에서 옵션 pivot_table 구하는 함수
def get_pivot_chain_within_group(raw_df, values = ['price', 'iv', 'delta']):

    # multiindex dataframe = [델타-행사가들 / 가격-행사가들, 등가, dte, 종가...] 생성

    res = pd.pivot_table(raw_df, values = values, index = raw_df.index, columns = ['strike'])
    aux = pd.pivot_table(raw_df, values = ['atm', 'dte', 'close'], index = raw_df.index)
    aux.columns = pd.MultiIndex.from_tuples([(col, "") for col in aux.columns])
    res = pd.concat([res, aux], axis = 1, join = 'inner')

    # 극내가옵션 거래안되서 데이터 NaN 나오는거 -> 콜풋 무관 내재가치로 맞춰놓기
    return res

# 옵션 pivot_table 에서 특정 trade 의 수익 구하는 함수!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# df : pivot table (index = 날짜 / columns = 행사가 / values = 가격)
# trade_dict = {'entry_date' : , 'k' : , 'number' : } 꼴

def generate_single_trade(df,
                   dist_from_atm : dict,
                   number_of_contracts : list,
                   preferred_weekday = 0,
                   dte_range = [35, 70]
                   ):

    '''
    return trade_dict with keys
    entry_date : one entry date in datetime format
    strikes : list of multiple ks
    number_of_contracts : trade size of each options in the same order as k
    '''

    # 0 = 월요일 ~ 6 = 일요일 on datetime.weekday()

    # 1)월요일 투자
    res = df.loc[filter(lambda x: datetime.weekday(x) == preferred_weekday, df.index)]

    # 2) 35~70 dte만 신규포지션
    res = res.loc[res['dte'].isin(range(dte_range[0], dte_range[1]))]

    # 3) strike price selection (if stated in relative distance then identify precise values)
    
    '''
    # 'number' : dist_from_atm 을 atm 대비 벌어진 값으로 설정되게
    # 'pct' : dist_from_atm 을 atm 대비 벌어진 수준(%) 구해서 알아서 설정되게
    # 'delta' : dist_from_atm 을 해당 시점 델타 기준으로 알아서 설정되게 (델타 20/15 -> 행사가 알아서 선정)

    ---> 위에 세개 전부 혼용 가능하도록 (진입시 "0.04%에 긋고 / +7.5 행사 위에다가 매도 후 / 델타 0.05짜리로 외가헤지" 와 같은 전략 구현)
    '''

    def find_strikes(row, dist_from_atm : dict, df = df):

        row = row.astype('float64')

        '''
        루프나 vectorize 할 요량이므로 dataframe 의 개별 행 (진입일자를 index 로 갖는) 에 대한 함수로 정의 
        어짜피 nested function 인 만큼 df 는 그냥 고정. 건드릴 이유 없음
        '''

        res = []

        for key in dist_from_atm.keys():

            if key == 'number':
                strike  = row['atm'].squeeze() + dist_from_atm.get(key)

            elif key == 'pct':
                raw_value = row['close'].squeeze() * (1 + dist_from_atm.get(key)) 
                strike = calc.get_closest_strike(raw_value)

            elif key == 'delta':
                strike = np.abs((row['delta'] - dist_from_atm.get(key))).astype('float64').idxmin()
            
            res.append(strike)

        return res

    
    res['trade_dict'] = res.apply(lambda row : {
        'entry_date' : row.name,
        'strikes' : find_strikes(row, dist_from_atm = dist_from_atm, df = df),
        'number_of_contracts' : number_of_contracts
        }, axis = 1)


    # 4) dynamic sizing
    return res['trade_dict'].tolist()

def get_single_trade_res(df, trade_dict: dict, is_complex_strat = False, profit_take = 0.5, stop_loss = 2):

    df = df['price'] # dataframe 전체에서 가격 부분만 사용

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
        profit_target = np.abs(df_pos_net_premium.iloc[0].squeeze()) * profit_take
        loss_target = - np.abs(df_pos_net_premium.iloc[0].squeeze()) * stop_loss
        
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
    trade_list = generate_single_trade(df, dist_from_atm = dist_from_atm, number_of_contracts = number_of_contracts, preferred_weekday = preferred_weekday, dte_range = dte_range)

    for trade in trade_list:
        trade_res = get_single_trade_res(df, trade, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['daily_ret']
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
    df = df_raw_subgroup.pipe(get_pivot_chain_within_group, values = ['price', 'iv', 'delta'])
    res = get_agg_return(df, dist_from_atm = dist_from_atm, number_of_contracts = number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)
    res = res['daily_ret'].cumsum()
    return res

#%%

if __name__ == "__main__":
    
    # 70-35일 남은 차월물 / 월요일마다 / 델타 0.1 / 양매도 / 중간 익손절 적용 (50% / -200%)

    put_dist_from_atm = {'delta' : 0.1}
    put_number_of_contracts = [-1]

    call_dist_from_atm = {'delta' : 0.1}
    call_number_of_contracts = [-1]
    
    preferred_weekday = 0
    dte_range = [35, 70]

    is_complex_strat = False
    profit_take = 0.5
    stop_loss = 2

    grouped_call = call_df_raw.groupby('expiry')
    grouped_put = put_df_raw.groupby('expiry')
    
# 테스트용 예시 : 2008-02-14 만기따리

    sample = grouped_call.get_group('2023-07-13')
    df = get_pivot_chain_within_group(sample)
    trade_list = generate_single_trade(df, call_dist_from_atm, call_number_of_contracts)
    ret = get_agg_return(df, call_dist_from_atm, call_number_of_contracts, is_complex_strat = False, profit_take = 0.5, stop_loss = 2)

    res0 = pd.concat([get_single_trade_res(df, trade_list[0], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_single_trade_res(df, trade_list[0], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')
    res1 = pd.concat([get_single_trade_res(df, trade_list[1], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_single_trade_res(df, trade_list[1], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')
    res2= pd.concat([get_single_trade_res(df, trade_list[2], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_single_trade_res(df, trade_list[2], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')
    res3 = pd.concat([get_single_trade_res(df, trade_list[3], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['df_premium'], get_single_trade_res(df, trade_list[3], is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)['cum_ret']], axis = 1, join = 'inner')

    res = get_final_result(sample, call_dist_from_atm, call_number_of_contracts, is_complex_strat = is_complex_strat, profit_take = profit_take, stop_loss = stop_loss)

#%%  실전 분석

result_put = grouped_put.apply(get_final_result, 
                dist_from_atm = put_dist_from_atm, 
                number_of_contracts = put_number_of_contracts, 
                preferred_weekday = preferred_weekday, 
                dte_range = dte_range,
                is_complex_strat = is_complex_strat, 
                profit_take = profit_take, 
                stop_loss = stop_loss)


result_call = grouped_call.apply(get_final_result, 
                dist_from_atm = call_dist_from_atm, 
                number_of_contracts = call_number_of_contracts, 
                preferred_weekday = preferred_weekday, 
                dte_range = dte_range)

