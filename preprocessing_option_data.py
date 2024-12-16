import pandas as pd
import sqlite3
import numpy as np
import option_calc

# table_name = 'weekly_thu'
# conn =sqlite3.connect("C:/Users/kwan/Desktop/option.db")
# query = f'SELECT * FROM {table_name}'
# chunks = pd.read_sql(query, conn, index_col = 'date', chunksize = 50000)
# data = pd.concat(chunks)

# import build_optiondb
# data = build_optiondb.get_data("20241031", '20241101', 'KRDRVOPK2I')
# table_name = 'monthly'

def process_raw_data(data, table_name):

    #3. 만기일(expiry), 잔존만기(dte : 오늘 - 만기일) 컬럼 만들기 -> 별도로 만기 계산해놔야함
    # 잔존만기는 만기당일 dte = 0.0001으로 떨어지게끔 나중에 그릭계산할때 별도 처리

    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.date.astype(str) # data의 인덱스 string 으로 변환

    conn = sqlite3.connect("C:/Users/kwan/Desktop/commonDB/db_option.db")

    exp_date = pd.read_sql(f"SELECT * FROM exp_{table_name}", conn, index_col = 'exp')
    data = data.merge(exp_date, how = 'inner', left_on = 'exp', right_index = True)

    data['dte'] = pd.to_datetime(data['exp_date']) - pd.to_datetime(data.index)
    data['dte'] = data['dte'].dt.days

    conn.close()

    #4. 당일 코스피200 ohlc / 당일 IV / 당일 할인금리 붙여서 컬럼 만들기

    conn = sqlite3.connect("C:/Users/kwan/Desktop/commonDB/db_timeseries.db")    

    k200 = pd.read_sql("SELECT * FROM k200", conn, index_col = 'date')
    k200.columns = k200.columns.str.cat(["_k200"] * len(k200.columns))
    data = data.merge(k200, how = 'left', left_index = True, right_index = True)
        
    vkospi = pd.read_sql("SELECT date, close FROM vkospi", conn, index_col = 'date')    
    vkospi.columns = vkospi.columns.str.cat(["_vkospi"] * len(vkospi.columns))
    data = data.merge(vkospi, how = 'left', left_index = True, right_index = True)

    base_rate = pd.read_sql("SELECT date, base_rate FROM rate_korea", conn, index_col = 'date') 
    base_rate.columns = ['rate']
    data = data.merge(base_rate, how = 'left', left_index = True, right_index = True)

    conn.close()

    #5. atm 및 moneyness 계산해놓기

    def find_closest_strike(x, interval = 2.5):
        divided = divmod(x, interval)
        if divided[1] >= 1.25: 
            result = divided[0] * interval + interval
        else:
            result = divided[0] * interval
        return result

    atm = k200['close_k200'].apply(find_closest_strike).rename('atm')
    data = data.merge(atm, how = 'left', left_index = True, right_index = True)
    
    # 콜풋 모두 외가를 양수로 치환

    sign_dummy = data['cp'].apply(lambda x : 1 if x == 'C' else -1)
    data['moneyness'] = (data['strike'] - data['atm']).multiply(sign_dummy)
    
    #6. adj_price
    # price == '-' 에 interp 밀어넣기 : 변동성은 거래소에서 제공한대로 직선보간된거 그대로 사용

    def calculate_additional_info(row):

        s = row['close_k200']
        k = row['strike']
        v = row['iv'] / 100
        t = (row['dte'] + 1) / 365 # 계산할때는 하루남은걸 dte = 2 / 만기당일인걸 dte = 1로 넣어서 계산
        r = row['rate'] / 100

        if row['cp'] == 'C':

            # 만기당일 에러 안나는 계산을 위해 dte = 0 이면 dte ~ 0.0001 같은 0 수렴값으로 바꾸기

            if row['dte'] == 0:
                t = 0.00001            
                price = np.maximum(row['close_k200'] - row['strike'], 0)        # dte ==0 이면 가격은 내재가치로 바꾸기
            
            else:
                price = np.maximum(np.round(option_calc.call_p(s, k, v, t, r), 2), 0.01)
        
            delta = np.round(option_calc.call_delta(s, k, v, t, r), 3)
            gamma = np.round(option_calc.gamma(s, k, v, t, r), 3)
            theta = np.round(option_calc.call_theta(s, k, v, t, r), 3)
            vega = np.round(option_calc.vega(s, k, v, t, r), 3)

        elif row['cp'] == 'P':

            # 만기당일 에러 안나는 계산을 위해 dte =0 이면 dte ~0값으로 바꾸기

            if row['dte'] == 0:
                t = 0.00001            
                price = np.maximum(row['strike'] - row['close_k200'], 0)        # dte ==0 이면 가격은 내재가치로 바꾸기
            
            else:
                price = np.maximum(np.round(option_calc.put_p(s, k, v, t, r), 2), 0.01)
        
            delta = np.round(option_calc.put_delta(s, k, v, t, r), 3)
            gamma = np.round(option_calc.gamma(s, k, v, t, r), 3)
            theta = np.round(option_calc.put_theta(s, k, v, t, r), 3)
            vega = np.round(option_calc.vega(s, k, v, t, r), 3)

        res = pd.Series([price, delta, gamma, theta, vega])

        return res
    
    data[['adj_price', 'delta', 'gamma', 'theta', 'vega']] = data.apply(calculate_additional_info, axis = 1)

    return data