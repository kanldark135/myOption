import pandas as pd
import requests
import json

date = '20180206'

url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

header = {
    "Content-Type": "text/html; charset=utf-8"
}

params = {
    "bld" : "dbms/MDC/STAT/standard/MDCSTAT12502",
    "locale" : "ko_KR",
    "trdDd" : date,
    "prodId" : "KRDRVOPK2I",
    "trdDdBox1" : date,
    "trdDdBox2" : "20240628",
    "mktTpCd" : "T",
    "rghtTpCd" : "T",
    "share" : 1,
    "money" : 3,
    "csvxls_isNo" : "false"
}

# response = requests.get(url = url, params = params, headers = header, data = json.dumps(params))
response = requests.get(url = url, params = params, headers = header)

if response.status_code == 200:
    raw_data = response.json()
    raw_data = raw_data['output']
else:
    print(f"Error code : {response.status_code}, failed to retrieve data")
