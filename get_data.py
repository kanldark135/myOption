import requests
import pandas as pd
from datetime import datetime as dt
import json
import sqlite3

date = '20230714'
today = '20240712'
product_id = 'KRDRVOPWKI'

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

    # 1. data의 인덱스는 당일날짜로 하기
    trade_date = pd.to_datetime(trade_date)
    data['date'] = trade_date
    data = data.set_index(['date'])

    #2. 종목명에서 정보 분리해내기
    data[['cp', '만기구분', 'strike']] = data['ISU_NM'].str.split(pat = " ", expand = True)[[1, 2, 3]]
    
    #3. 만기, 잔존만기 컬럼 만들기
    
    #4. 외부값 (당일 코스피200 종가 / 당일 할인금리) 붙여서 컬럼 만들기
    #5. moneyness 계산해놓기
    #6. 당일 그릭값 계산해서 각 컬럼 만들기

    return data

# 7. db에 저장
# def stack_data(data):

#     conn = sqlite3.connect("C:/Users/kanld/Desktop/option.db")
#     cur = conn.cursor()

#     create_table_query = '''
#     CREATE TABLE IF NOT EXISTS weekly_thu (
#         ISU_CD TEXT,
#         ISU_SRT_CD TEXT,
#         ISU_NM TEXT,
#         TDD_CLSPRC REAL,
#         FLUC_TP_CD INTEGER,
#         CMPPREVDD_PRC REAL,
#         TDD_OPNPRC REAL,
#         TDD_HGPRC REAL,
#         TDD_LWPRC REAL,
#         IMP_VOLT REAL,
#         NXTDD_BAS_PRC REAL,
#         ACC_TRDVOL INTEGER,
#         ACC_TRDVAL INTEGER,
#         ACC_OPNINT_QTY INTEGER,
#         SECUGRP_ID TEXT,
# )

# '''
