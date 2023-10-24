
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta


# custom strategy 저장

# def apply_ta(df):
#     bb_20 = df.ta.bbands(20, 2)
#     bb_60 = df.ta.bbands(60, 2)
#     sslow_533 = df.ta.stoch(5, 3, 3)
#     rsi_14 = df.ta.rsi(14)
#     rsi_signal = rsi_14.rolling(window = 6).mean()
#     psar = df.ta.psar(0.02, 0.02, 0.2)
    
#     res = pd.concat([df, bb_20, bb_60, sslow_533, rsi_14, rsi_signal, psar], axis = 1)
#     res.columns = res.columns.str.lower()
#     res = res.loc[:, (~res.columns.str.startswith(('bbb', 'bbp')))] # 필요없는 컬럼 삭제


#     return res
# df = apply_ta(k200)


# 1. 과열 침체 역방향 시그널

class contrarian:

    def __init__(self, df):
        self.df = df

    def through_bbands(self, length = 20, std = 2):
        
        res = pd.DataFrame(index = self.df.index, columns = ['signal'])
        bbands = self.df.ta.bbands(length, std)
        bbands.columns = bbands.columns.str.lower()
        bbands = bbands.loc[:, (~bbands.columns.str.startswith(('bbb', 'bbp')))] # 필요없는 컬럼 삭제
        
        # 롱 시그널
        cond_long = (self.df['close'] < bbands['bbl_' + str(length) + "_" + str(float(std))]) 
        res.loc[cond_long, 'signal'] = 1
        # 숏 시그널
        cond_short = (self.df['close'] > bbands['bbu_' + str(length) + "_" + str(float(std))]) 
        res.loc[cond_short, 'signal'] = -1

        return res
    
    def stoch_rebound(self, k = 5, d = 3, smooth_d = 3):

        res = pd.DataFrame(index = self.df.index, columns = ['signal'])
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch = stoch.reindex(res.index)
        stoch.columns = stoch.columns.str.lower()
        stoch = stoch.rename(columns = {f'stochk_{k}_{d}_{smooth_d}' : 'k', f'stochd_{k}_{d}_{smooth_d}' : 'd'}) 

        # 롱 시그널
        cond_long_1 = stoch['k'].shift(1) <= 20 # K가 전날 20 밑에 (오늘은 상관 없음)
        cond_long_2 = stoch['k'] > stoch['d'] # K가 오늘 D를 상향돌파
        cond_long = cond_long_1 * cond_long_2
        res.loc[cond_long, 'signal'] = 1

        # 숏 시그널
        cond_short_1 = stoch['k'].shift(1) > 80 # K가 전날 80 위에 (오늘은 상관 없음)
        cond_short_2 = stoch['k'] < stoch['d'] # K가 오늘 D를 하향돌파
        cond_short = cond_short_1 * cond_short_2
        res.loc[cond_short_1 * cond_short, 'signal'] = -1

        return res

    def rsi_rebound(self, length = 14, scalar = 100):

        res = pd.DataFrame(index = self.df.index, columns = ['signal'])
        rsi = self.ta.rsi(length = length, scalar = scalar)

        # 롱 시그널
        

        return None

class notrade:
        
    df_vkospi = pd.read_pickle("./data_pickle/df_vkospi.pkl")
    df_vix = pd.read_pickle("./data_pickle/df_vix.pkl")

    def vix_curve_invert(self, inversion_scale = 0):
        
        # 추가보완여지 : slope index 하락할때는 X / 상승반전시*에는 notrade 조건에서 빼기
        # 상승반전을 뭘로 정의할건지?
        
        res = self.df_vix.loc[self.df_vix['slope_index'] < inversion_scale].index
        return res

    def vkospi_above_n(self, high_or_close = 'close', quantile = 0.8):

        if high_or_close == 'high':

            limit = self.df_vkospi['high'].quantile(quantile)
        else:
            limit = self.df_vkospi['close'].quantile(quantile)

        res = self.df_vkospi.loc[self.df_vkospi[high_or_close] > limit].index
        return res
    
# 2. 정추세 지속 시그널

# 3. squeeze 폭발 시그널

# 4. 주가 / 지표 다이버전스

class divergence:

    def __init__(self, df):
        self.df = df
