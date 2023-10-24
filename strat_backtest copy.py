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

    aux = pd.pivot_table(raw_df, values = ['atm', 'dte', 'close'], index = raw_df.index)
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
    
    res = res.tolist()
    return res

def get_trade_result(df_pivoted, trade_dict: dict):

    ''' trade_dict
    entry_date : datetime
    trade : list
    contract : list
    '''
    try:
        df_trade_area = df_pivoted.loc[trade_dict['entry_date'] : , trade_dict['trade']]
        df_net_premium = df_trade_area.multiply(np.negative(trade_dict['contract']), axis = 1)
        df_ret = (df_net_premium.shift(1) - df_net_premium).fillna(0)
        df_cumret = df_ret.cumsum()
        daily_ret = df_ret.sum(axis = 1)
        cumret = df_cumret.sum(axis = 1)

        res = {
            'area' : df_trade_area,
            'df_premium' : df_net_premium,
            'df_ret' : df_ret,
            'df_cumret' : df_cumret,
            'daily_ret' : daily_ret,
            'cumret' : cumret
        }
    # 트레이드가 아예 없는 경우 발생하는 에러 처리
    except (IndexError, TypeError):
        df_trade_area = pd.Series(0, index = df_pivoted.index)
        df_net_premium = pd.Series(0, index = df_pivoted.index)        
        df_ret = pd.Series(0, index = df_pivoted.index)
        df_cumret = pd.Series(0, index = df_pivoted.index)
        daily_ret = pd.Series(0, index = df_pivoted.index)
        cumret = pd.Series(0, index = df_pivoted.index)

        res = {
            'area' : df_trade_area,
            'df_premium' : df_net_premium,
            'df_ret' : df_ret,
            'df_cumret' : df_cumret,
            'daily_ret' : daily_ret,
            'cumret' : cumret
        }
        
    # 옛날데이터의 경우 행사가가 없어서 산정해놓은 행사가에 매칭되는 자료가 없는 경우 -> 그냥 안한 셈 치기

    except KeyError:
        df_trade_area = pd.Series(0, index = df_pivoted.index)
        df_net_premium = pd.Series(0, index = df_pivoted.index)        
        df_ret = pd.Series(0, index = df_pivoted.index)
        df_cumret = pd.Series(0, index = df_pivoted.index)
        daily_ret = pd.Series(0, index = df_pivoted.index)
        cumret = pd.Series(0, index = df_pivoted.index)

        res = {
            'area' : df_trade_area,
            'df_premium' : df_net_premium,
            'df_ret' : df_ret,
            'df_cumret' : df_cumret,
            'daily_ret' : daily_ret,
            'cumret' : cumret
        }

    return res

def stop_trade(trade_result : dict, is_complex_strat = False, profit_take = 0.5, stop_loss = -2):
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

    # 익손절 구현
    if is_complex_strat == True:
        
        profit_target = profit_take
        loss_target = stop_loss
    else:
        profit_target = max(np.abs(initial_premium) * profit_take, 0.01)
        loss_target = min(np.abs(initial_premium) * stop_loss, -0.01)
        
    # IndexError 발생상황 1 : 중간청산이 안 되는경우 (익절 또는 손절 안되고 그대로 만기까지 가는 경우) : liquidate_date = 만기로 설정
    try:
        liquidate_date = cumret[
            (cumret >= profit_target)|
            (cumret <= loss_target)
        ].index[0]

    except IndexError:
        liquidate_date = None

    df_trade_area = trade_result['area'].loc[:liquidate_date]
    df_net_premium = trade_result['df_premium'].loc[:liquidate_date] 
    df_ret = trade_result['df_ret'].loc[:liquidate_date]
    df_cumret = trade_result['df_cumret'].loc[:liquidate_date]
    daily_ret = trade_result['daily_ret'].loc[:liquidate_date]
    cumret = trade_result['cumret'].loc[:liquidate_date]

    res = {
    'area' : df_trade_area,
    'df_premium' : df_net_premium,
    'df_ret' : df_ret,
    'df_cumret' : df_cumret,
    'daily_ret' : daily_ret,
    'cumret' : cumret
    }

    return res

def get_agg_return(df_pivoted, entry_dates, trade_spec, dte_range = [35,70], is_complex_strat = False, profit_take = 0.5, stop_loss = -2):

    '''한 만기 내에서 모든 진입시점 만들고 / 각 진입에 대한 만기까지의 손익 및 / 중간익손절까지 반영하여 => 
    각 매매의 결과 (=result_list) list / 전부 합산한 해당 만기의 일일손익 output'''

    res_list = []
    trade_list = create_trade_entries(df_pivoted, entry_dates, trade_spec, dte_range)
    for trade in trade_list:
        trade_res = stop_trade(get_trade_result(df_pivoted, trade),
                is_complex_strat = is_complex_strat, 
                stop_loss = stop_loss,
                profit_take = profit_take)['daily_ret']
        res_list.append(trade_res)
    try:
        daily_ret = pd.concat(res_list, ignore_index = True, axis = 1).sum(axis = 1)
    except ValueError:
        daily_ret = pd.Series(0, index = df_pivoted.index)
    res = {
        'result_list' : res_list,
        'daily_ret' : daily_ret
    }
    return res

def get_final_result(df_raw, 
                     entry_dates, 
                     trade_spec,  
                     dte_range = [35,70], 
                     is_complex_strat = False, 
                     profit_take = 0.5, 
                     stop_loss = -2):

    df = df_raw.pipe(get_pivot_table, values = ['adj_price', 'iv_interp', 'delta'])
    res = get_agg_return(df, entry_dates, trade_spec, dte_range, is_complex_strat, profit_take, stop_loss)
    res = res['daily_ret']

    return res

#%%

if __name__ == "__main__":

    data_from = '2010-01-15' # 옛날에는 행사가가 별로 없어서 전략이 이상하게 나감

    # myentrydate / exitdate : 온갖 방법으로 진입 /청산시점 구할 것이기 때문에 어떤 함수로 방법론을 특정할 수가 없음
    # 어디선가 진입 /청산 df 구해와서 그 datetimeIndex 만 가져온다고 치고 거기서부터 시작하기

    #1. 스토캐스틱 과열 기준 진입일
    import get_entry_date
    k200 = pd.read_pickle("./data_pickle/k200.pkl")
    ta_based_entry = get_entry_date.contrarian(k200)
    stoch = ta_based_entry.stoch_rebound(k = 5, d = 3, smooth_d = 3)
    entry_dates_stoch = stoch.loc[stoch['signal'] == 1].index

    # 2. 목요일마다 진입 (월 = 0 ~ 일 = 6)
    entry_dates_thursday = (monthly.index.unique()[monthly.index.unique().weekday == 3])

    # # naked call
    # trade_spec = {'C': [('delta', -0.10, -1)]}

    # naked put
    trade_spec = {'P': [('pct', -0.10, -1)]}

    # # strangle
    # trade_spec = {'C': [('pct', 0.08, -1)], 
    #               'P': [('pct', -0.08, -1)]}

    # # iron condor
    # trade_spec = {'C': [('pct', 0.08, -1), ('pct', 0.09, 1)], 
    #               'P': [('pct', -0.08, -1), ('pct', -0.09, 1)]}

    # # put 111
    # trade_spec = {'P': [('delta', -0.25, 1), ('delta', -0.21, -1), ('delta', -0.05, -1)]}

    dte_range = [35, 70]

    is_complex_strat = False
    profit_take = 0.5
    stop_loss = -2

    grouped = monthly.loc[data_from:].groupby('expiry')

    all_expiry = grouped.groups.keys()

# 테스트용 예시 : 2008-02-14 만기따리

    sample = grouped.get_group('2020-04-09')
    sample_pivoted = sample.pipe(get_pivot_table)

    trade_list = create_trade_entries(df_pivoted = sample_pivoted, 
                                      entry_dates = entry_dates_stoch, 
                                      trade_spec = trade_spec)
    
    single_result_list = []
    for trade in trade_list:
        res = stop_trade(get_trade_result(sample_pivoted, trade_dict = trade))
        single_result_list.append(res)
    
    ret = get_agg_return(df_pivoted = sample_pivoted, 
                        entry_dates = entry_dates_stoch, 
                        trade_spec = trade_spec,
                        is_complex_strat = is_complex_strat, 
                        profit_take = profit_take, 
                        stop_loss = stop_loss)
    
    result = get_final_result(sample,
                        entry_dates = entry_dates_stoch, 
                        trade_spec = trade_spec,
                        is_complex_strat = is_complex_strat, 
                        profit_take = profit_take, 
                        stop_loss = stop_loss
                        )

#%%  실전 분석

alltime_backtest = grouped.apply(get_final_result, 
                        entry_dates = entry_dates_stoch, 
                        trade_spec = trade_spec,
                        is_complex_strat = is_complex_strat, 
                        profit_take = profit_take, 
                        stop_loss = stop_loss)
