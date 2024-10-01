import requests
import pandas as pd
import numpy as np
import json
import time
import datetime
import sqlite3
import preprocessing_data

# trade_date = '20240924'
# today = '20240925'
# product_id = 'KRDRVOPWKI' 

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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
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

    # 데이터타입 좀 더 효율적으로 바꾸기

    # volume 쪽 바꾸기
    def str_to_int(df_series):
        df = df_series.astype(str).str.replace(",", "") # 1) str 포맷의 콤마 없애기
        df = pd.to_numeric(df, errors = 'coerce')
        df = df.fillna(0).astype("int64")

        return df
    
    data.index = pd.to_datetime(data.index).date
    # 인덱스를 datetime 이 아니라 string 형태로 일단 저장해야 나중에 indexing 할때 편해보임
    # datetime 으로 바꾸는건 나중에 작업할 때 파이썬에 불러와서 수정

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

def save_to_db(data, product_id, db_path = "C:/Users/kanld/Desktop/option.db"):

    product_id_placeholder = {
                        'KRDRVOPK2I' : 'monthly',
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'}
    
    table_name = product_id_placeholder[product_id]
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    data.to_sql(table_name, conn, if_exists = "append", index = True)

    conn.close()

def save_multiple_dates(start_date, end_date, today, db_path = "C:/Users/kanld/Desktop/option.db"):

    date_range = pd.date_range(start_date, end_date)
    product_id_placeholder = {
                        'KRDRVOPK2I' : 'monthly'
                        # 'KRDRVOPWKI' : 'weekly_thu',
                        # 'KRDRVOPWKM' : 'weekly_mon'
                        }
    
    for product_id in product_id_placeholder.keys():
        start_time = time.time()

        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            data = get_data(date_str, today, product_id)

            if data is None:
                print(f"{date_str} 데이터 없어서 저장 안하고 패스")
                continue

            data = preprocessing_data.process_raw_data(data, product_id_placeholder[product_id])
            save_to_db(data, product_id, db_path)
            print(f"{product_id_placeholder[product_id]} 테이블에 {date_str} 날짜 데이터 저장 완료")
        end_time = time.time()
        print(f"{product_id} 전부 불러오는데 {end_time - start_time} 초 걸림")


if __name__ == "__main__":

    start_date = '20240801'
    end_date = '20240930'
    today = '20240930' # 주말이나 휴일에 조회할때는 직전영업일로
    # today = datetime.datetime.today().strftime("%Y%m%d")
    db_path = "C:/Users/kanld/Desktop/option.db"

    save_multiple_dates(start_date, end_date, today, db_path)