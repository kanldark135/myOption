import sqlite3
import pandas as pd
import numpy as np
import datetime

#%% 
# path_categorical = "./data_categorical.xlsx"


# db_categorical = 

# categorical data
    # sheet_names = pd.ExcelFile(database).sheet_names

    # conn = sqlite3.connect(db_path)
    # cur = conn.cursor()

    # if database == path_categorical:

    #     for name in sheet_names:    
    #         df = pd.read_excel(database, name, index_col = 0, dtype = 'str')
    #         df.to_sql(name, conn, if_exists = "replace", index = True)
    

path_timeseries = "./data_timeseries.xlsx"
db_timeseries = "C:/Users/kwan/Desktop/commonDB/db_timeseries.db"

conn = sqlite3.connect(db_timeseries)
cur = conn.cursor()

sheet_names = pd.ExcelFile(path_timeseries).sheet_names

for name in sheet_names:
    df = pd.read_excel(path_timeseries, name, index_col = 0)
    first_empty_col = np.argmax(df.columns.str.contains("Unnamed"))
    
    if first_empty_col != 0:
        df = df.iloc[:, 0 : first_empty_col]
    
    df = df.dropna(how = 'all', axis = 1)
    df = df.sort_index(ascending = True)
    df.index = df.index.map(lambda x : datetime.datetime.strftime(x, "%Y-%m-%d"))
    
    df.to_sql(name, conn, if_exists = "replace", index = True)

conn.close()
# %%
