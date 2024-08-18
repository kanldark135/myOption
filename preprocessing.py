

#%% 

# 옵션가격 월물 업데이트 (매달마다 월물은 만기 2달 전 1일부터 ~ 만기까지 (만기 껴있는 달 20일까지))
# fnguide 엑셀가격 기준

import pandas as pd
import numpy as np
from datetime import datetime as dt

#%% 

xlsx_file = "./raw_data/외부데이터종합.xlsx"

df_k200 = pd.read_excel(xlsx_file, sheet_name = "k200_hts", usecols = "A:E")
df_k200.columns = ['date', 'open', 'high', 'low', 'close']
df_k200.set_index(['date'], inplace = True)
df_k200.sort_index(ascending = True, inplace = True)
df_k200.to_pickle("./working_data/df_k200.pkl")

df_vkospi = pd.read_excel(xlsx_file, sheet_name = "vkospi_hts", usecols = "A:E")
df_vkospi.columns = ['date', 'open', 'high', 'low', 'close']
df_vkospi.set_index(['date'], inplace = True)
df_vkospi.sort_index(ascending = True, inplace = True)
df_vkospi.to_pickle("./working_data/df_vkospi.pkl")

df_vix = pd.read_excel(xlsx_file, sheet_name = "vix_bbg", usecols = "A:O", skiprows = 3, index_col = 0)
df_vix.to_pickle("./working_data/df_vix.pkl")

df_rate = pd.read_excel(xlsx_file, sheet_name = "rate_infomax", index_col = 0)
df_rate.to_pickle("./working_data/df_rate.pkl")


#%% 
this_year = 2023

pre = pd.read_pickle("./raw_data/raw_monthly.pkl")
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
final.to_pickle("./raw_data/raw_monthly.pkl")


#%% 위클리 업데이트

import pandas as pd
import numpy as np
from datetime import datetime as dt

this_year = "2023_w"

pre = pd.read_pickle("./raw_data/raw_weekly.pkl")

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
final.to_pickle("./raw_data/raw_weekly.pkl")

# %%

import pandas as pd
import numpy as np
import option_calc as calc

# # pandas does not provide explicit extrapolation with its built-in df.interpolate. hence make custom
# # interpolation /extrapolation cubic spline function using scipy.interpolate library.

# from scipy.interpolate import CubicSpline

# def custom_extrapolation(df):
#     bool_not_na = df['iv'].notna()
#     valid_index = df.loc[bool_not_na, 'strike'].values
#     valid_value = df.loc[bool_not_na, 'iv'].values.astype('float64')

#     interpolator = CubicSpline(valid_index, valid_value)
#     res = interpolator(df['strike'])

#     df['iv_interp'] = res

#     return df

df_monthly = pd.read_pickle("./raw_data/raw_monthly.pkl")
df_weekly = pd.read_pickle("./raw_data/raw_weekly.pkl")
df_kospi = pd.read_pickle("./working_data/df_k200.pkl")
df_vkospi = pd.read_pickle("./working_data/df_vkospi.pkl")['close']
df_base_rate = pd.read_pickle("./working_data/df_base_rate.pkl")


def create_table(df_raw, df_kospi, df_vkospi, df_base_rate):

# 1, 2, 3, 4. 주변값들 : 당일 등가격 / 코스피 / vkospi / 기준금리 (계산용) 같다 붙이기

    def find_closest_strike(x, interval = 2.5):
        divided = divmod(x, interval)
        if divided[1] >= 1.25: 
            result = divided[0] * interval + interval
        else:
            result = divided[0] * interval
        return result

    df_atm = df_kospi['close'].apply(find_closest_strike).rename('atm')
    df_m_1 = df_raw.merge(df_atm, how = 'left', left_index = True, right_index = True)
    df_m_2 = df_m_1.merge(df_kospi, how = 'left', left_index = True, right_index = True)
    df_m_3 = df_m_2.merge(df_vkospi, how = 'left', left_index = True, right_index = True)
    df_m_4 = df_m_3.merge(df_base_rate, how = 'left', left_index = True, right_index = True).rename(columns = {'close_x' : 'close', 'close_y' : 'vkospi'})

#4. 등가격 대비 괴리도 (콜풋 모두 외가격을 양수로 치환)

    def moneyness(cp, strike, atm):
        if cp == "C":
            res = strike - atm
        elif cp == "P":
            res = atm - strike
        return res

    df_m_4['moneyness'] = list(map(moneyness, df_m_4['cp'], df_m_4['strike'], df_m_4['atm']))

#5~6. 내재변동성 컬럼으로 빼서 로우 축소

    price = df_m_4[df_m_4['title'] == '종가']
    df_m_5 = df_m_4[df_m_4['title'] == '내재변동성']

    df_m_6 = df_m_5.merge(price['value'], how = 'left', left_on = [df_m_5.index, df_m_5.cp, df_m_5.expiry, df_m_5.strike], right_on = [price.index, price.cp, price.expiry, price.strike])

    df_m_6 = df_m_6.rename(columns = {'key_0' : 'date', 'value_x' : 'iv', 'value_y' : 'price'})
    df_m_6.drop(columns = ['key_1', 'key_2', 'key_3'], inplace = True)
    df_m_6 = df_m_6.set_index('date')
    df_m_6 = df_m_6.drop(columns = ['title'])
    df_m_6['expiry'] = pd.to_datetime(df_m_6['expiry'], errors = 'coerce')

# 7 IV 보간 및 보간된 IV 기반의 model_price 산정해서 nan 또는 맥락없는 0 값에 껴넣기
    # 12월 배당락 걸쳐있는 차년도 만기 옵션들의 경우 
    # 매년마다 예상배당수익률 계산하는거 개 낭비같아서 그냥 0.03으로 픽스시킴

    grouped = df_m_6.groupby(by = ['expiry', 'date', 'cp'])

    # interpolate ivs

    def iv_interpolation(df):
        res = df['iv']\
        .interpolate(method = 'spline', order = 2)\
        .fillna(method = 'bfill')\
        .fillna(method = 'ffill')
        df['iv_interp'] = res
        return df

    df = grouped.apply(iv_interpolation)
    df.index = df.index.droplevel([0, 1, 2])

    # interpolate price computed from interpolated iv

    df['div_yield'] = 0.03 * (~(df.index.year == df['expiry'].dt.year))  # 매년 dividend yield 3% 였던걸로 대충 퉁 침
    call_mask = df['cp'] == 'C'
    df['price_interp'] = np.where(call_mask, calc.call_p(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield), calc.put_p(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield))

    ## interp 가격 -> 
    # price nan/0에 interp 밀어넣기 / 0.01 이하 가격 0.01로 통일시키기 / dte = 1일때는 아예 내재가치로 바꾸기
        
    df['adj_price'] = df['price'].mask((df['price'].isna())|(df['price'].eq(0)), df['price_interp'])
    df['adj_price'] = df['adj_price'].mask(df['adj_price'].lt(0.01), 0.01)
    df['adj_price'] = df['adj_price'].mask((df['dte'] == 1) & (df['cp'] == "C"), np.maximum(df['close'] - df['strike'], 0))
    df['adj_price'] = df['adj_price'].mask((df['dte'] == 1) & (df['cp'] == "P"), np.maximum(df['strike'] - df['close'], 0))
    
    # greeks computed from interpolated iv

    df['delta'] = np.where(call_mask, calc.call_delta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield), calc.put_delta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield))
    df['gamma'] = calc.gamma(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield)
    df['theta'] = np.where(call_mask, calc.call_theta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield), calc.put_theta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield))
    df['vega'] = calc.vega(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield)

    return df

monthly = create_table(df_monthly, df_kospi, df_vkospi, df_base_rate)
weekly = create_table(df_weekly, df_kospi, df_vkospi, df_base_rate)
weekly = weekly.loc[weekly['dte'] < 9]
monthly['id'] = monthly['cp'] + monthly['expiry'].dt.strftime('%Y%m').astype('str') + monthly['strike'].astype('str')
weekly['id'] = weekly['cp'] + weekly['expiry'].dt.strftime('%Y%m%d').astype('str') + weekly['strike'].astype('str')


#%% pkl

monthly.to_pickle("./working_data/df_monthly.pkl")
weekly.to_pickle("./working_data/df_weekly.pkl")


 # %% sql

import sqlite3

local_file_path = "./option_k200.db"

def db_connect(local_file_path):
        
    conn = sqlite3.connect(local_file_path)
    return conn

def perform_sql(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)

create_table = '''CREATE TABLE IF NOT EXISTS weekly_total (
                date DATE,
                cp TEXT,
                expiry DATE,
                iv REAL,
                dte INT,
                atm REAL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                vkospi REAL,
                rate REAL,
                moneyness REAL,
                price REAL,
                iv_interp REAL,
                div_yield REAL,
                price_interp REAL,
                adj_price REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                id TEXT PRIMARY KEY
                );
    '''

if __name__ == "__main__":
    conn = db_connect(local_file_path)
    perform_sql(conn, create_table)

    monthly.to_sql("weekly_total", conn, if_exists = 'replace', index = True)

    conn.commit()
    conn.close()

