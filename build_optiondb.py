import requests
import pandas as pd
import numpy as np
import json
import time
import datetime
import sqlite3
import duckdb
import preprocessing_option_data
import pathlib
import get_all_backtests
import backtest as bt

product_id_placeholder = {'KRDRVOPK2I' : 'monthly',
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'
}

def get_data(trade_date:str, today:str, product_id:str):
    # Define the URL
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    # Define the parameters
    params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT12502',
        'locale': 'ko_KR',
        'trdDd': trade_date,
        'prodId': product_id,
        'trdDdBox1': trade_date,
        'trdDdBox2': today,
        'mktTpCd': 'T',
        'rghtTpCd': 'T',
        'share': 1,
        'money': 3,
        'csvxls_isNo': 'false'
    }

    # Define the headers
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'http://data.krx.co.kr',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201050101',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }

    # Make the request
    response = requests.post(url, data=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        print(f"{trade_date} data is called")
        data = response.json()
        data = data['output']
        data = pd.DataFrame(data)
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

    # Check if data is empty (휴무일 등등으로 거래 없는 날)

    if data.empty:
        print(f"{trade_date} 는 휴무일 등으로 값이 없음")
        return None
    
    # 1. data의 인덱스는 당일날짜로 하기
    trade_date = pd.to_datetime(trade_date)
    data['date'] = trade_date
    data = data.set_index(['date'])

    #2. 필요한 컬럼만 추리고 컬럼명 변경
    name_placeholder = {'ISU_CD' : 'code', 
                        'ISU_NM' : 'name', 
                        'TDD_CLSPRC' : 'close',
                        'TDD_OPNPRC' : 'open',
                        'TDD_HGPRC' : 'high',
                        'TDD_LWPRC' : 'low',
                        "IMP_VOLT" : 'iv',
                        'ACC_TRDVOL' : 'trd_volume',
                        "ACC_TRDVAL" : 'trd_value',
                        'ACC_OPNINT_QTY' : 'open_interest'
                        }
    
    data = data[name_placeholder.keys()].rename(columns = name_placeholder)

    #3. 종목명에서 정보 분리해내기
    data[['cp', 'exp', 'strike']] = data['name'].str.split(pat = " ", expand = True)[[1, 2, 3]]

    # 1) 데이터타입 좀 더 공간효율적으로 / 2) preprocessing_option_data 의 변수로 주기 위한 데이터타입 변환

    # volume 쪽 바꾸기
    def str_to_int(df_series):
        df = df_series.astype(str).str.replace(",", "") # 1) str 포맷의 콤마 없애기
        df = pd.to_numeric(df, errors = 'coerce')
        df = df.fillna(0).astype("int64")

        return df
    
    data['exp'] = data['exp'].astype(str).str.replace(".0", "") # exp도 일부러 문자형으로 변환하되 소수점만 제거

    data['strike'] = data['strike'].astype('float64') 

    data['close'] = pd.to_numeric(data['close'], errors = 'coerce').astype('float64').round(2)
    data['open'] = pd.to_numeric(data['open'], errors = 'coerce').astype('float64').round(2)
    data['high'] = pd.to_numeric(data['high'], errors = 'coerce').astype('float64').round(2)
    data['low'] = pd.to_numeric(data['low'], errors = 'coerce').astype('float64').round(2) 
    data['iv'] = pd.to_numeric(data['iv'], errors = 'coerce').astype('float64').round(2)

    data['trd_volume'] = str_to_int(data['trd_volume'])
    data['trd_value'] = str_to_int(data['trd_value'])
    data['open_interest'] = str_to_int(data['open_interest'])

    return data

def save_to_db(data, product_id, db_path):

    product_id_placeholder = {
                        'KRDRVOPK2I' : 'monthly',
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'}
    
    table_name = product_id_placeholder[product_id]
    
# sqlite
    conn = sqlite3.connect(db_path + 'db_option.db')
    data.to_sql(table_name, conn, if_exists = "append", index = True, index_label = 'date')
    conn.close()

# duckdb     
    duckdb_conn = duckdb.connect(db_path + 'option.db')
    data = data.reset_index()
    data = data.rename(columns = {'index' : 'date'}) #  duckdb 기준으로는 따로 to_sql 에서 df 에 index 있는지 유무 식별 불가하므로 전부 컬럼화
    duckdb_conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM data")
    duckdb_conn.close()

def save_multiple_dates(start_date, end_date, today, db_path):

    date_range = pd.date_range(start_date, end_date)
    product_id_placeholder = {
                        'KRDRVOPK2I' : 'monthly',
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'
                        }
    
    for product_id in product_id_placeholder.keys():
        start_time = time.time()

        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            data = get_data(date_str, today, product_id)

            if data is None:
                print(f"{date_str} 데이터 없어서 저장 안하고 패스")
                continue

            data = preprocessing_option_data.process_raw_data(data, product_id_placeholder[product_id])
            save_to_db(data, product_id, db_path)
            print(f"{product_id_placeholder[product_id]} 테이블에 {date_str} 날짜 데이터 저장 완료")
        end_time = time.time()
        print(f"{product_id} 전부 불러오는데 {end_time - start_time} 초 걸림")


def resave_data():

    db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
    option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")
    df_k200 = bt.get_timeseries(db_path, "k200")['k200']

    def get_table_weekly(table, cp, term = 1):

        result = pd.DataFrame()

        entry_dates = dict(
            mon = df_k200.weekday(0),
            tue = df_k200.weekday(1),
            wed = df_k200.weekday(2),
            thu = df_k200.weekday(3),
            fri = df_k200.weekday(4)
        )

        table = [table]
        types = ['moneyness']
        term = list(get_all_backtests.generate_iterables([term]))
        ref_values = dict(
            moneyness = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40] # 위클리용
        )

        leg_numbers = 0
        offset_values = dict(
            moneyness = get_all_backtests.generate_moneyness_offset(number_offset_legs = leg_numbers, interval_between_legs=2.5, min_offset=2.5, max_offset = 15)
        )

        query_vars = get_all_backtests.generate_query_vars(entry_dates,
                                        table,
                                        types,
                                        term,
                                        ref_values,
                                        offset_values,
                                        offset_type = ['moneyness&moneyness'])

        for entry, table, type, term, ref_value, offset_value in query_vars['moneyness&moneyness']:
        
            entry_date = entry_dates.get(entry)

            leg1 = {'entry_dates' : entry_date,
                        'table' : table,
                        'cp' : cp,
                        'type' : type,
                        'select_value' : ref_value,
                        'term' : term[0],
                        'volume' : 1,
                        'dte' : [1, 999],
                        'iv_range' : [0, 999]
                        }
            
            df = bt.backtest(leg1).concat_df
            res = df.loc[df.index.get_level_values(0) == df.index.get_level_values(1)].loc[:, (slice(None), ['iv', 'dte', 'adj_price', 'delta', 'gamma', 'vega', 'close_k200'])]
            res = res.droplevel('date', axis = 0).droplevel(level = 0, axis = 1)
            
            prefix = "_".join([str(entry), str(table), str(ref_value)])
            res.columns = pd.MultiIndex.from_product([[prefix], res.columns])

            result = pd.concat([result, res], axis = 1, join = 'outer')

        result = result.sort_index()

        return result

    # 통상적인 진입시점의 IV 테이블 -> 옵션의 모든 IV 테이블이 아님...
    def process_table(table, cp, term = 1):

        df = get_table_weekly(table, cp, term)
        df = df.reset_index()
        df = df.melt(id_vars = [('entry_date', '')], ignore_index = False)
        df.columns = ['entry_date', 'variable', 'items', 'value']
        df = df.set_index('entry_date')

        split = df['variable'].str.split("_", expand = True)

        if table in ['weekly_mon', 'weekly_thu']:

            df['weekday'] = split[0].replace({"mon": 0, "tue" : 1, "wed" : 2, "thu" : 3, "fri" : 4})
            df['moneyness'] = split[3].astype(float)
            df['table_name'] = split[1] + "_" + split[2]
        
        elif table == 'monthly':
            df['weekday'] = split[0].replace({"mon": 0, "tue" : 1, "wed" : 2, "thu" : 3, "fri" : 4})
            df['moneyness'] = split[2].astype(float)       
            df['table_name'] = table

        # dte 추가 

        df_dte = df.loc[df['items'] == 'dte'][['value', 'variable']]
        df_k200 = df.loc[df['items'] == 'close_k200'][['value', 'variable']]
        df_rest = df.loc[~(df['items'] == 'dte')&~(df['items'] == 'close_k200')]

        df = pd.merge(df_rest, df_dte, how = 'inner', left_on = [df_rest.index, df_rest['variable']], right_on = [df_dte.index, df_dte['variable']])
        df = pd.merge(df, df_k200, how = 'inner', left_on = ['key_0', 'key_1'], right_on = [df_k200.index, df_k200['variable']])

        # 정리
        df = df.set_index(['key_0'])
        df = df.drop(['variable', 'variable_x', 'variable_y'], axis = 1)
        df = df.rename(columns = {'key_1': 'name', 'value_x' : 'value', 'value_y' : 'dte', 'value' : 'k200'})
        df.index.rename('date', inplace = True)
        df['term'] = term
        df['cp'] = cp

        df = df.dropna(how = 'all', subset = ['dte', 'k200'])

        df.index = df.index.strftime("%Y-%m-%d")

        return df

    call_mon = process_table("weekly_mon", "C")
    call_thu = process_table("weekly_thu", "C")
    put_mon = process_table("weekly_mon", "P")
    put_thu = process_table("weekly_thu", "P")
    df = pd.concat([call_mon, call_thu, put_mon, put_thu], axis = 0)
    df = df.reset_index()

    conn = duckdb.connect(option_path)
    conn.execute("DROP TABLE weekly_data")
    conn.execute("CREATE TABLE weekly_data AS SELECT * FROM df")

    call_mon = process_table("monthly", "C", 1)
    put_mon = process_table("monthly", "P", 1)
    call_mon_back = process_table("monthly", "C", 2)
    put_mon_back = process_table("monthly", "P", 2)
    df = pd.concat([call_mon, call_mon_back, put_mon, put_mon_back], axis = 0)
    df = df.reset_index()

    conn = duckdb.connect(option_path)
    conn.execute("DROP TABLE monthly_data")
    conn.execute("CREATE TABLE monthly_data AS SELECT * FROM df")
    conn.close()

if __name__ == "__main__":

# 1. data_timeseries.xlsx 에서 db_timeseries.db 로 시장데이터 업데이트
    import build_findatadb
    build_findatadb.dump_from_xlsx()

# 2. 이거 하기 전에 exp_monthly / exp_weekly 에서 지나간 월물 만기들 수기로 기입 필요

# 3. start_date 부터 end_date 까지 필요한 데이터들 거래소에서 scrape 해서 option.db monthly / weekly_mon / weekly_thu 테이블에 적재
    start_date = '20250401'
    end_date = '20250430'
    today = '20250502' # 주말이나 휴일에 조회할때는 직전영업일로
    db_path = "C:/Users/kwan/Desktop/commonDB/"
    save_multiple_dates(start_date, end_date, today, db_path)

#4. 적재된 option 로데이터들 가지고 통상적으로 필요한 pivot_table 형태로 재가공해서 resave 하기 (monthly_data / weekly_data 테이블에 적재)

    time.sleep(5)
    resave_data()