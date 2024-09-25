import requests
import pandas as pd
from datetime import datetime as dt
import json
import time
import sqlite3

trade_date = '20240924'
today = '20240925'
product_id = 'KRDRVOPWKI' 

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


    #3. 만기일(expiry), 잔존만기(dte : 오늘 - 만기일) 컬럼 만들기 -> 별도로 만기 계산해놔야함
    #4. 당일 코스피200 ohlc / 당일 할인금리 붙여서 컬럼 만들기
    #5. moneyness 계산해놓기
    #6. 거래안되서 가격없는애들 -> 거래소데이터에는 IV는 있으므로 interpolation 은 불필요 / 그냥 있는 iv에 adj_price 구하기
    #7. 당일 그릭값 계산해서 각 컬럼 만들기

    return data

def save_to_db(data, product_id, path = "C:/Users/kanld/Desktop/option.db"):

    product_id_placeholder = {'KRDRVOPK2I' : 'monthly',
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'}
    
    table_name = product_id_placeholder[product_id]
    
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    data.to_sql(table_name, conn, if_exists = "append", index = True)

    conn.close()


start_date = '20240801'
end_date = '20240924'

def save_multiple_dates(start_date, end_date, today):

    date_range = pd.date_range(start_date, end_date)
    product_id_placeholder = {
                        'KRDRVOPWKI' : 'weekly_thu',
                        'KRDRVOPWKM' : 'weekly_mon'}
    
    for product_id in product_id_placeholder.keys():
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            data = get_data(date_str, today, product_id)

            if data is None:
                print(f"{date_str} 데이터 없어서 저장 안하고 패스")
                continue

            save_to_db(data, product_id)
            print(f"{product_id_placeholder[product_id]} 테이블에 {date_str} 날짜 데이터 저장 완료")