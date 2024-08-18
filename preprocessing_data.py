import pandas as pd
import sqlite3
import numpy as np


conn =sqlite3.connect("C:/Users/kanld/Desktop/option.db")
query = 'SELECT * FROM monthly'
df = pd.read_sql_query(query, con = conn)
df = df.set_index(['date'])
df.index = pd.to_datetime(df.index)

# 외부데이터 날짜 / 만기에 맞게 조인

mkt_data = pd.read_excel("./mkt_data_infomax.xlsx", usecols = "A:F", header = 0, index_col = 0)
df = df.merge(mkt_data[['rate_30d', 'k200', 'vkospi']], how = 'left', left_index = True, right_index = True)

maturity = pd.read_excel("./mkt_data_infomax.xlsx", usecols = "H:I", header = 0, index_col = 0)
df = df.merge(maturity, how = 'left', left_on = ['exp'], right_index = True)


conn.close()
