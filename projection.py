# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:18:39 2023

@author: Kwan
"""

import option_calc
import pandas as pd
import xarray as xr
import numpy as np
import scipy.stats as scistat
import scipy.optimize as sciop



# 실제 IV 및 옵션가격 추출 (웹기반은 아니고 HTS 복사 -> 옵션종합.xlsx 에 붙여넣기 후 저장 -> 이후 실행)

weekly = pd.read_excel("C:/Users/문희관/Desktop/종합.xlsx", sheet_name = "data_weekly", skiprows = [0], usecols = "A, I, J, L, T").drop_duplicates('행사가').set_index('행사가').set_axis(['call_iv', 'call_p', 'put_p', 'put_iv'], axis = 1)
front_month = pd.read_excel("C:/Users/문희관/Desktop/종합.xlsx", sheet_name = "data_front_month", skiprows = [0], usecols = "A, I, J, L, T").drop_duplicates('행사가').set_index('행사가').set_axis(['call_iv', 'call_p', 'put_p', 'put_iv'], axis = 1)
back_month = pd.read_excel("C:/Users/문희관/Desktop/종합.xlsx", sheet_name = "data_back_month", skiprows = [0], usecols = "A, I, J, L, T").drop_duplicates('행사가').set_index('행사가').set_axis(['call_iv', 'call_p', 'put_p', 'put_iv'], axis = 1)

iv_weekly = weekly[['call_iv', 'put_iv']]
iv_front_month = front_month[['call_iv', 'put_iv']]
iv_back_month = back_month[['call_iv', 'put_iv']]


## dictionary of functions that I would constantly refer to

tenor_df = {1 : iv_weekly, 2 : iv_front_month, 3: iv_back_month}
call_function = {'price': option_calc.call_p, 'delta': option_calc.call_delta, 'theta': option_calc.call_theta, 'vega': option_calc.vega, 'gamma': option_calc.gamma}
put_function = {'price': option_calc.put_p, 'delta': option_calc.put_delta, 'theta': option_calc.put_theta, 'vega': option_calc.vega, 'gamma': option_calc.gamma}


# 개별옵션의 IV 의 변화 Breakdown

# 1) curve 자체가 horizontal 하게 움직인다는 가정 : 외가옵션이 등가로 들어오면 현재 등가 vol 수준으로 조정 (통상 upward skew 되어있는 시장에서는 하락 요인)
# 2) curve 의 vertical 이동 : 시장 전체 변동성 동일 수준만큼 확대/축소
# 3) curvature 의 변화 : 커브의 모양 자체가 변화

# vol curve generate 하는 함수_2 : 현재 vol 이 주가가 해당 구간까지 움직일 시 그대로 realized 된다는 가정

def interpolate_iv(s_list, df):
    
    new_index = df.index.union(s_list)
    
    df_new = df.reindex(new_index)
    
    df_new.interpolate(inplace = True)

    return df_new


'''1) 개별 행사가(=종목) 에 대한 projection'''

def single_strike(strike, tenor, s, t, r, v = None, callput = 'call'):  
    
    # 월물구분 : tenor 참조하여 사용하는 df 판단, v : None 이면 엑셀파일에서 불러온 현재 IV 가져다가 사용, 아니면 직접 대입
    
    s_list = [s + 1.25 * i for i in range(-20, 20)]
    time_passed = [7 * i for i in range(int(t / 7) + 1)]
    result_dict = dict()
    
    ## 나중에 vol curve 적용하는 구조로 가게 되면 v를 1개 값이 아니라 S 변화에 따라 같이 pair 된(dict 구조?) input 으로 재수정해야함
    
    # 1. 행사가별 시장 볼 그대로 unchanged 된다는 가정시 
    
    if v == None:
    
        if callput == 'call':
            
            v = tenor_df[tenor].loc[strike, 'call_iv'] / 100
        
            for func_name in call_function.keys():
            
                df_null = pd.DataFrame(index = time_passed, columns = s_list)
            
                for time in time_passed:
                    for spot in s_list:
                        df_null.at[time, spot] = call_function[func_name](spot, strike, v, (t - time)/365, r)
                        df_null.index.name = 'time_passed'
                        df_null.columns.name = 'spot'
                        df_null.mask(pd.isna(df_null), 0, inplace = True)
                        result_dict[func_name] = df_null
                        
        else:
            
            v = tenor_df[tenor].loc[strike, 'put_iv'] / 100
            
            for func_name in put_function.keys():
            
                df_null = pd.DataFrame(index = time_passed, columns = s_list)
            
                for time in time_passed:
                    for spot in s_list:
                        df_null.at[time, spot] = put_function[func_name](spot, strike, v, (t - time)/365, r)
                        df_null.index.name = 'time_passed'
                        df_null.columns.name = 'spot'
                        df_null.mask(pd.isna(df_null), 0, inplace = True)
                        result_dict[func_name] = df_null
    
    # 2, 3) 각각 현재 시장
    
    elif v != None:
        
        if callput == 'call':
        
            for func_name in call_function.keys():
            
                df_null = pd.DataFrame(index = time_passed, columns = s_list)
            
                for time in time_passed:
                    for spot in s_list:
                        df_null.at[time, spot] = call_function[func_name](spot, strike, v, (t - time)/365, r)
                        df_null.index.name = 'time_passed'
                        df_null.columns.name = 'spot'
                        df_null.mask(pd.isna(df_null), 0, inplace = True)
                        result_dict[func_name] = df_null
                        
        else:
            
            for func_name in put_function.keys():
            
                df_null = pd.DataFrame(index = time_passed, columns = s_list)
            
                for time in time_passed:
                    for spot in s_list:
                        df_null.at[time, spot] = put_function[func_name](spot, strike, v, (t - time)/365, r)
                        df_null.index.name = 'time_passed'
                        df_null.columns.name = 'spot'
                        df_null.mask(pd.isna(df_null), 0, inplace = True)
                        result_dict[func_name] = df_null

    return result_dict


'''2) 여러 행사가(=종목) 과 수량을 조합한 특정 전략에 대한 projection 뽑아내는 함수'''





# 행사가(종목) 구분 + 각 종목별 월물 구분 + 실제 종목별 매매수량 -> 콜따로 풋따로

def single_strategy(s, r, call_strikes = None, call_tenor = None, call_n = None, put_strikes = None, put_tenor = None, put_n = None, call_iv = None, put_iv = None, call = 0, put = 0):
    
    ## 콜 부분 따로 계산
    
    if call == 0:
        
        call_sum = 0
    
    else:
 
        call_i = 0
        call_sum = 0   
        call_dummy = dict()
        
        for call_strike in call_strikes:
            
            dte = dte_by_tenor[call_tenor[call_i] - 1]
             
            ds_call = call_n[call_i] * xr.Dataset(single_strike(call_strike, call_tenor[call_i], s, dte, r, callput = 'call'))
            
            call_dummy[call_strike] = ds_call
            
            # 첫 ds_call 은 first dataset 으로 사용
            
            if call_i == 0:
            
                call_sum = ds_call
                
            else:
                
                call_sum = call_sum + ds_call
            
            call_i = call_i + 1
    
    # 풋 부분 따로 계산
    
    if put == 0:
        
        put_sum = 0
    
    else:
    
        put_i = 0
        put_sum = 0   
        put_dummy = dict()
    
        for put_strike in put_strikes:
             
            dte = dte_by_tenor[put_tenor[put_i] - 1]
            
            ds_put = put_n[put_i] * xr.Dataset(single_strike(put_strike, put_tenor[put_i], s, dte, r, callput = 'put'))
            
            put_dummy[put_strike] = ds_put
            
            # 첫 ds_put 은 first dataset 으로 사용
            
            if put_i == 0:
            
                put_sum = ds_put
                
            else:
                
                put_sum = put_sum + ds_put
            
            put_i = put_i + 1
            
    # 콜,풋 합산 및 df로 전환
    
    sum_total = call_sum + put_sum
    
    current_price = float(sum_total.sel(time_passed = 0, spot = s)['price']) # 진입시 가격
    
    
    sum_total = sum_total.assign(pnl = sum_total['price'] - current_price)
    
    price = sum_total['price'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    pnl = sum_total['pnl'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    max_profit_atm = pnl.loc[:, s].max() # 등가에서 최대 수익
    delta = sum_total['delta'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    gamma = sum_total['gamma'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    theta = sum_total['theta'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    vega = sum_total['vega'].to_dataframe().unstack(['spot']).droplevel(0, axis = 1)
    
    return current_price, max_profit_atm, pnl, delta, gamma, theta, vega
    



'''3. 위 함수들을 여러 행사가 조합에 걸쳐 동시에 돌려서 optimal 행사가 조합 찾기

1) 적정 Buffer 확보되는 가운데 = 이건 통계적으로 분석해서 행사가별 distance 는 사전적으로 정의하는 수 밖에
2) ATM 부근에서 그래도 깔고 앉아있으면 Credit 좀 확보되는 행사가 조합을 = target credit 제시
    optimization 으로 역산'''


# 아예 사전 정의되어야 할 시장 data (현재주가, 상품별 잔존만기, 금리)


s = 320 # 현재주가
dte_by_tenor = [7, 14, 49] # 만기상품별로 남은 일수 t
r = 0.037 # 현재금리


# 변동성은 1) single_strike 함수 내에 정의된 대로, 위에 시장데이터에서 가져오던가 2) 별도의 변동성 dataframe 으로 정의해놓고 single_strike 함수에 feed
# 행사가 = 개별 상품이므로 아래에서 선택

'''주요 전략들은 dict 형태로 사전에 만들어놓기'''

# 이하 입력 내용은 UI로 구현하더라도 사람이 직접 입력해야 하는 사항들인 건 동일

# 1. 사용하는 상품 행사가 정의

# 전략의 시작점

call_start_strike = 330
put_start_strike = 310

# 전략

call_distance_set = [0, 2.5, 20, 40] 
call_tenor = [3, 3, 3, 3]
call_n = [100, -100, -120, 120]


put_distance_set = [0, 7.5, 22.5] # 20-20 dist broken wing condor
put_tenor = [3, 3, 3]
put_n = [40, -80, 40]





# =============================================================================
# call_distance_set = [0, 5] # weekly 5pt width iron condor
# put_distance_set = [0, 5]
# 
# # 2. 사용하는 상품 만기 정의
# 
# call_tenor = [1,1]
# put_tenor = [1,1]
# 
# # 3. 사용하는 상품별 수량 정의
# 
# call_n = [-100, 100]
# put_n = [-100, 100]
# =============================================================================


def strategy_loop(call_start = None, put_start = None, loop_n = 1):
    
    '''loop_n : number of looping over 2.5 incremental strike prices. stick to 1 if you want to return the result at single strike price'''
    
    data_dict = dict()
    
    global single_strategy

    if call_start == None:
        
        call = 0
        put = 1
        
        put_start_list = np.array([put_start - i * 2.5 for i in range(loop_n)])
        put_strikes_set = [[put_start - j for j in put_distance_set] for put_start in put_start_list]
        

        
        for k in range(loop_n):

            result = single_strategy(s, r, put_strikes = put_strikes_set[k], put_tenor = put_tenor, put_n = put_n, call = call, put = put)
            
            data_dict[2.5 * k] = result
        
    elif put_start == None:
        
        call = 1
        put = 0
        
        call_start_list = [call_start + i * 2.5 for i in range(loop_n)]
        call_strikes_set = [[call_start + j for j in call_distance_set] for call_start in call_start_list]
        
        for k in range(loop_n):

            result = single_strategy(s, r, call_strikes = call_strikes_set[k], call_tenor = call_tenor, call_n = call_n, call = call, put = put)
            
            data_dict[2.5 * k] = result
        
    else:
        
        call = 1
        put = 1
        
        call_start_list = [call_start + i * 2.5 for i in range(loop_n)]
        call_strikes_set = [[call_start + j for j in call_distance_set] for call_start in call_start_list]
        
        put_start_list = [put_start - i * 2.5 for i in range(loop_n)]
        put_strikes_set = [[put_start - j for j in put_distance_set] for put_start in put_start_list]
        
        
        for k in range(loop_n):

            result = single_strategy(s, r, call_strikes = call_strikes_set[k], call_tenor = call_tenor, call_n = call_n, put_strikes = put_strikes_set[k], put_tenor = put_tenor, put_n = put_n, call = call, put = put)
            
            data_dict[2.5 * k] = result

    ### 논리 구성이, 어짜피 콜풋 둘중 하나는 있을것이므로 콜이 없으면 풋 / 풋이 없으면 콜 / 둘다 있으면 둘다
    ### 둘다 없는 경우는 어짜피 아예 이 함수 안 쓰고 있을 것...
        
    return data_dict
        

# =============================================================================
# result = strategy_loop(call_start_strike, put_start_strike, 5)
# 
# # initial_credit/debit
# 
# list_price = [result[i][0] for i in result.keys()]
#                 
#         
# =============================================================================


def sort_result(result_dict):
    
    current_price = dict()
    
    max_profit_atm = dict()
    
    pnl = dict()
    
    delta = dict()
    
    gamma = dict()
    
    theta = dict()
    
    vega = dict()
    
    
    for i in result_dict.keys():
        
        current_price[i] = result_dict[i][0]
        
        max_profit_atm[i] = result_dict[i][1]
        
        pnl[i] = result_dict[i][2]
        
        delta[i] = result_dict[i][3]
        
        gamma[i] = result_dict[i][4]
        
        theta[i] = result_dict[i][5]
        
        vega[i] = result_dict[i][6]
    
    return current_price, max_profit_atm,  pnl, delta, gamma, theta, vega
    
    

    


