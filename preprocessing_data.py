import pandas as pd
import sqlite3
import numpy as np
import option_calc

conn =sqlite3.connect("C:/Users/kanld/Desktop/option.db")
query = 'SELECT * FROM monthly'


def process_raw_data(conn, table_name):

    query = f'SELECT * FROM {table_name}'

    df = pd.read_sql_query(query, con = conn, index_col = ['date'])
    df.index = pd.to_datetime(df.index)

    #3. (완료) 만기일(expiry), 잔존만기(dte : 오늘 - 만기일) 컬럼 만들기 -> 별도로 만기 계산해놔야함
    # 잔존만기는 만기당일 dte = 0.0001으로 떨어지게끔 나중에 그릭계산할때 별도 처리

    exp_date = pd.read_excel("./aux_data.xlsx", sheet_name = table_name, usecols = 'A:B', index_col = 'exp')
    df = df.merge(exp_date, how = 'inner', left_on = 'exp', right_index = True)
    df['dte'] = df['exp_date'] - df.index
    df['dte'] = df['dte'].dt.days

    #4. 당일 코스피200 ohlc / 당일 IV / 당일 할인금리 붙여서 컬럼 만들기
    
    k200 = pd.read_excel("./aux_data.xlsx", sheet_name = 'k200', usecols = 'A:E', index_col = 'date')
    df = df.merge(k200, how = 'left', left_index = True, right_index = True)
        
    iv = pd.read_excel("./aux_data.xlsx", sheet_name = 'iv', usecols = 'A,E', index_col = 'date')
    df = df.merge(iv, how = 'left', left_index = True, right_index = True)

    short_rate = pd.read_excel("./aux_data.xlsx", sheet_name = 'rate', usecols = 'A:B', index_col = 'date')
    df = df.merge(short_rate, how = 'left', left_index = True, right_index = True)

    #5. atm 및 moneyness 계산해놓기

    def find_closest_strike(x, interval = 2.5):
        divided = divmod(x, interval)
        if divided[1] >= 1.25: 
            result = divided[0] * interval + interval
        else:
            result = divided[0] * interval
        return result

    atm = k200['close_k200'].apply(find_closest_strike).rename('atm')
    df = df.merge(atm, how = 'left', left_index = True, right_index = True)
    
    # 콜풋 모두 외가를 양수로 치환

    sign_dummy = df['cp'].apply(lambda x : 1 if x == 'C' else -1)
    df['moneyness'] = (df['strike'] - df['atm']).multiply(sign_dummy)
    
    #6. adj_price
    
    # price == '-' 에 interp 밀어넣기 : 변동성은 거래소에서 제공한대로 직선보간된거 그대로 사용

    df_call = df.loc[df['cp'] == 'C']
    df['price_interp'] = df.loc[call_mask]np.where(call_mask, option_calc.call_p(df.close_k200, 
                                                                df.strike, 
                                                                df.iv/100,
                                                                (df.dte + 1)/365, # 중요! : Pricing 할때 +1일 해서 실시 / 만기 당일 종가는 내재가치로 날리기
                                                                df.rate/100),                     
                                            option_calc.put_p(df.close, 
                                                              df.strike, 
                                                              df.iv/100, 
                                                              (df.dte + 1)/365, 
                                                              df.rate/100)
    )
     
    df['adj_price'] = df['close']
        
    df['adj_price'] = df['price'].mask((df['price'].isna())|(df['price'].eq(0)), df['price_interp'])
    df['adj_price'] = df['adj_price'].mask(df['adj_price'].lt(0.01), 0.01)
    # dte = 0일때는 아예 내재가치로 바꾸기
    df['adj_price'] = df['adj_price'].mask((df['dte'] == 0) & (df['cp'] == "C"), np.maximum(df['close'] - df['strike'], 0))
    df['adj_price'] = df['adj_price'].mask((df['dte'] == 0) & (df['cp'] == "P"), np.maximum(df['strike'] - df['close'], 0))
    
    # greeks computed from interpolated iv

    df['delta'] = np.where(call_mask, calc.call_delta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield), calc.put_delta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield))
    df['gamma'] = calc.gamma(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield)
    df['theta'] = np.where(call_mask, calc.call_theta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield), calc.put_theta(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield))
    df['vega'] = calc.vega(df.close, df.strike, df.iv_interp, df.dte/365, df.rate/100, df.div_yield)

    
    #7. 당일 그릭값 계산해서 각 컬럼 만들기



conn.close()
