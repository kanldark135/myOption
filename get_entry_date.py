import talib
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

k200 = pd.read_pickle('./data_pickle/k200.pkl')

# custom strategy 저장

def myContrarian(df):
    bb_20 = df.ta.bbands(20, 2)
    bb_60 = df.ta.bbands(60, 2)
    sslow_533 = df.ta.stoch(5, 3, 3)
    rsi_14 = df.ta.rsi(14)
    rsi_signal = rsi_14.rolling(window = 6).mean()
    psar = df.ta.psar(0.02, 0.02, 0.2)
    
    res = pd.concat([df, bb_20, bb_60, sslow_533, rsi_14, rsi_signal, psar], axis = 1)
    res.columns = res.columns.str.lower()

    return res

ta.stoch