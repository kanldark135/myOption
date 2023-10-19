#%%

# 옵션가격 월물 업데이트 (매달마다 월물은 만기 2달 전 1일부터 ~ 만기까지 (만기 껴있는 달 20일까지))
# fnguide 엑셀가격 기준

import pandas as pd
import numpy as np
from datetime import datetime as dt

this_year = 2023

pre = pd.read_pickle("./data_pickle/monthly.pkl")
pre = pre[pre['expiry'] < '2023-01-01'] # 당해년도는 삭제후 재업

df = pd.read_excel(f"C:/Users/kanld/Desktop/{this_year}.xlsx", skiprows = 7, index_col = 0)

# 불필요한 데이터 정리
df.columns = df.loc['Name']
df.index.name = 'date'
df.drop(df.index[0:5], axis = 0, inplace = True)

# 종목 - 만기매칭 Series 따로 생성
bool_expiry = df.loc['D A T E'].isin(['만기일'])
df_expiry = df.iloc[3].loc[bool_expiry].apply(lambda x : dt.strptime(str(x), "%Y%m%d"))

# callput / expiry / strike / 속성값으로 구성된 multiindex 생성

expiry = dict()
callput = dict()
strike = dict()

# 컬럼명 (종목명) 에서 필요한 데이터들 각각 Parsing 해서 뽑아내기 :

for i in df.columns:
    expiry[i] = df_expiry[i]
    ii = i.split(" ")
    callput[i] = str(ii[1])
    strike[i] = float(ii[3])

dummy = pd.DataFrame([expiry, callput, strike]).T.rename(columns = {0 : 'expiry', 1: 'cp', 2 : 'strike'})

df2 = df.copy().T
df2 = df2.join(dummy)
df2 = df2.set_index(['expiry', 'cp', 'strike', 'D A T E']).T

idx = pd.IndexSlice
df2 = df2.loc[:, idx[:, :, :, ~bool_expiry]] # 만기일 column 은 전부 제거
df2.index = pd.to_datetime(df2.index)

df_result = df2.melt(ignore_index = False)

# DTE 계산
df_result = df_result.assign(dte = (df_result.expiry - df_result.index) / pd.Timedelta(1, "D") + 1)

# 1) 잔존일수가 존재하는 (아직 살아있는) 종목들만 추리기
df_result = df_result.loc[df_result['dte'] >= 1]

# 2) 0제거
post = df_result[df_result.value != 0]
post = post.rename(columns = {'D A T E' : 'title'})

# 최종적으로 원래 있던거에 둘이 합치고 완전히 똑같은 로우는 지우기
final = pd.concat([pre, post], axis = 0)
final = final.drop_duplicates()

# pickle 파일로 다시 빼기
final.to_pickle("./data_pickle/monthly.pkl")


#%% 위클리 업데이트

import pandas as pd
import numpy as np
from datetime import datetime as dt

this_year = "2023_w"

pre = pd.read_pickle("./data_pickle/weekly.pkl")

df = pd.read_excel(f"C:/Users/kanld/Desktop/{this_year}.xlsx", skiprows = 7, index_col = 0)

# 불필요한 데이터 정리
df.columns = df.loc['Name']
df.index.name = 'date'
df.drop(df.index[0:5], axis = 0, inplace = True)

# 종목 - 만기매칭 Series 따로 생성
bool_expiry = df.loc['D A T E'].isin(['만기일'])
df_expiry = df.iloc[3].loc[bool_expiry].apply(lambda x : dt.strptime(str(x), "%Y%m%d"))

# callput / expiry / strike / 속성값으로 구성된 multiindex 생성

expiry = dict()
callput = dict()
strike = dict()

# 컬럼명 (종목명) 에서 필요한 데이터들 각각 Parsing 해서 뽑아내기 :

for i in df.columns:
    expiry[i] = df_expiry[i]
    ii = i.split(" ")
    callput[i] = str(ii[1])
    strike[i] = float(ii[3])

dummy = pd.DataFrame([expiry, callput, strike]).T.rename(columns = {0 : 'expiry', 1: 'cp', 2 : 'strike'})

df2 = df.copy().T
df2 = df2.join(dummy)
df2 = df2.set_index(['expiry', 'cp', 'strike', 'D A T E']).T

idx = pd.IndexSlice
df2 = df2.loc[:, idx[:, :, :, ~bool_expiry]] # 만기일 column 은 전부 제거
df2.index = pd.to_datetime(df2.index)

df_result = df2.melt(ignore_index = False)

# DTE 계산
df_result = df_result.assign(dte = (df_result.expiry - df_result.index) / pd.Timedelta(1, "D") + 1)

# 1) 잔존일수가 존재하는 (아직 살아있는) 종목들만 추리기
df_result = df_result.loc[df_result['dte'] >= 1]

# 2) 0제거
post = df_result[df_result.value != 0]
post = post.rename(columns = {'D A T E' : 'title'})

# 최종적으로 원래 있던거에 둘이 합치고 완전히 똑같은 로우는 지우기
final = pd.concat([pre, post], axis = 0)
final = final.drop_duplicates()

# pickle 파일로 다시 빼기
final.to_pickle("./data_pickle/weekly.pkl")

# %%

import pandas as pd
import numpy as np
import option_calc

df_monthly = pd.read_pickle("./data_pickle/monthly.pkl")
df_weekly = pd.read_pickle("./data_pickle/weekly.pkl")
df_kospi = pd.read_pickle("./data_pickle/k200.pkl")

def preprocessing(df_option, df_kospi):

    # 현재주가에 Closest 한 행사가 찾기 함수

    def find_closest_strike(x, interval = 2.5):
        divided = divmod(x, interval)
        if divided[1] >= 1.25: 
            result = divided[0] * interval + interval
        else:
            result = divided[0] * interval
        return result


    find_closest_strike = np.vectorize(find_closest_strike)

    # 일단 변동성이랑 종가랑 같은 테이블에 있는거 => 테이블 둘로 쪼갰다가 다시 merge 하는방법밖에 없어보임
    # (unstacking 은 안되고, pivot으로 하면 멀티컬럼 생성되버림)

    # 콜 풋도 만기 당일 내재가치 처리해줄려면 다시 빼서 각각 계산해줘야하는데 
    # 테이블 전체에 iterrow:if row['cp] == 'call' or 'put' 으로 돌리는것보다 그냥 쪼개서 처리하는게 효율적이라는 생각    
    # 결국 콜_p/ 콜_v / 풋_p/ 풋_v 로 각각 계산한다음 콜이랑 풋 따로 테이블 빼서 쓰는 식으로 하기로

    # 1) df_kospi join

    df_append = df_kospi.assign(atm = find_closest_strike(df_kospi['close']))

    # 2) ATM 대비 Moneyness 표기

    df_c = df_option[df_option['cp']== "C"]
    df_c = df_c.merge(df_append, how = 'left', left_index = True, right_index = True)
    df_c = df_c.assign(moneyness = df_c.strike - df_c.atm)

    df_p = df_option[df_option['cp']== "P"]
    df_p = df_p.merge(df_append, how = 'left', left_index = True, right_index = True)
    df_p = df_p.assign(moneyness = df_p.atm - df_p.strike)

    # 3) 매일마다 존재하는 옵션 근/차월물 식별하여 레이블링
    ## 여기 문제 있는게, 아예 상장조차 되지 않은 물건들도 raw data 의 형식으로 인해 마치 있는것처럼 처리되어있음
    # 해서 특히 weekly 의 경우 당연히 없어야 할 back / backback 월물이 있는것처럼 되어있는데 일단 NA값으로만 처리


    def grouping(df):

        grouped = df.groupby(by = df.index)
        keys = grouped.groups.keys()
        dummy_df = pd.DataFrame()

        def cycle_indexing(part_df):
            index_table = part_df[['expiry']].drop_duplicates().sort_values(by = ['expiry'], axis = 0)
            index_table['cycle'] = np.arange(len(index_table))
            index_table = index_table.set_index('expiry')

            res = part_df.merge(index_table['cycle'], how = 'left', left_on = part_df['expiry'], right_index = True)
            
            return res
        
        for key in keys:
            part_df = grouped.get_group(key)
            res = cycle_indexing(part_df)
            dummy_df = pd.concat([dummy_df, res], ignore_index = False)

        return dummy_df

    df_c = df_c.pipe(grouping)
    df_p = df_p.pipe(grouping)

    # 4) 최종적으로 콜가격/콜변동성/풋가격/풋변동성 4개의 dataframe 생성

    df_cv = df_c[df_c['title'] == '내재변동성']
    df_cp = df_c[df_c['title'] == '종가']

    # 만기당일은 종가 -> 행사가 Payoff 로 교체
    df_cp['value'] = df_cp['value'].mask(df_cp.index == df_cp['expiry'], np.maximum(df_cp['close'] - df_cp['strike'], 0))

    # 풋
    df_pv = df_p[df_p['title'] == '내재변동성']
    df_pp = df_p[df_p['title'] == '종가']
    
    df_pp['value'] = df_pp['value'].mask(df_pp.index == df_pp['expiry'], np.maximum(df_pp['strike'] - df_pp['close'], 0))

    # 다시 같이 합치기 

    df_put = df_pv.merge(df_pp['value'], how = 'left', left_on = [df_pv.index, df_pv.cycle, df_pv.strike, df_pv.dte], right_on = [df_pp.index, df_pp.cycle, df_pp.strike, df_pp.dte]).set_index('key_0')
    df_call = df_cv.merge(df_cp['value'], how = 'left', left_on = [df_cv.index, df_cv.cycle, df_cv.strike, df_cv.dte], right_on = [df_cp.index, df_cp.cycle, df_cp.strike, df_cp.dte]).set_index('key_0')
    df_put.drop(['key_1', 'key_2', 'key_3'], axis = 1, inplace = True)
    df_call.drop(['key_1', 'key_2', 'key_3'], axis = 1, inplace = True)

    df_put.rename(columns = {'value_x' : "iv", 'value_y' : 'price'}, inplace = True)
    df_call.rename(columns = {'value_x' : "iv", 'value_y' : 'price'}, inplace = True)

    # 기준금리 concat

    base_rate = pd.read_pickle("./data_pickle/base_rate.pkl")

    df_put = df_put.merge(base_rate, how = 'left', left_index = True, right_index = True)
    df_call = df_call.merge(base_rate, how = 'left', left_index = True, right_index = True)

    # 그릭 : 그릭은 크게 다르지 않은 숫자로 판단
    # 배당락 미반영 : 해당 만기에 배당락있어도 배당수익률 미반영한숫자임 (이거까지 하기에는...)
    # 그럼에도 IV기준 실효 그릭을 구할수 있는데가 없어서 내가 직접 계산... (거래소 데이터는 HV 기준임)

    df_call['iv'] = df_call['iv'].fillna(0)
    df_put['iv'] = df_put['iv'].fillna(0)

    df_put = df_put.assign(
        delta = option_calc.put_delta(df_put.close, df_put.strike, df_put.iv, df_put.dte/365, df_put.rate/100),
        gamma = option_calc.gamma(df_put.close, df_put.strike, df_put.iv, df_put.dte/365, df_put.rate/100),
        theta = option_calc.put_theta(df_put.close, df_put.strike, df_put.iv, df_put.dte/365, df_put.rate/100),
        vega = option_calc.vega(df_put.close, df_put.strike, df_put.iv, df_put.dte/365, df_put.rate/100) 
    )
    
    df_call = df_call.assign(
        delta = option_calc.call_delta(df_call.close, df_call.strike, df_call.iv, df_call.dte/365, df_call.rate/100),
        gamma = option_calc.gamma(df_call.close, df_call.strike, df_call.iv, df_call.dte/365, df_call.rate/100),
        theta = option_calc.call_theta(df_call.close, df_call.strike, df_call.iv, df_call.dte/365, df_call.rate/100),
        vega = option_calc.vega(df_call.close, df_call.strike, df_call.iv, df_call.dte/365, df_call.rate/100) 
    )

    return df_call, df_put

if __name__ == "__main__":

    call_month, put_month = preprocessing(df_monthly, df_kospi)
    call_week, put_week = preprocessing(df_weekly, df_kospi)
    
    call_month.to_pickle("./data_pickle/call_monthly.pkl")
    put_month.to_pickle("./data_pickle/put_monthly.pkl")
    call_week.to_pickle("./data_pickle/call_weekly.pkl")
    put_week.to_pickle("./data_pickle/put_weekly.pkl")

 # %%
