import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/kanld/Desktop/vkospi.xlsx", usecols = 'A:E', header = 0, index_col =  0)

df['전일종가_오늘종가'] = df['종가'] - df['종가'].shift(-1)
df['전일종가_오늘고가'] = df['고가'] - df['종가'].shift(-1)

df['2일종가_오늘종가'] = df['종가'] - df['종가'].shift(-1).rolling(2).min().shift(-1)
df['2일종가_오늘고가'] = df['고가'] - df['종가'].shift(-1).rolling(2).min().shift(-1)

df['3일종가_오늘종가'] = df['종가'] - df['종가'].shift(-1).rolling(3).min().shift(-2)
df['3일종가_오늘고가'] = df['고가'] - df['종가'].shift(-1).rolling(3).min().shift(-2)

def find_quantile(df, col_name, top_n = 10):
    df_50 = df[col_name].quantile(0.5)
    df_80 = df[col_name].quantile(0.8)
    df_95 = df[col_name].quantile(0.95)
    df_99 = df[col_name].quantile(0.99)
    
    top_10 = df[col_name].loc[df[col_name].rank(ascending = False) < top_n].sort_values(ascending = False)

    return df_50, df_80, df_95, df_99, top_10