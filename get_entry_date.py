
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

k200 = pd.read_pickle('./working_data/df_k200.pkl')
vkospi = pd.read_pickle('./working_data/df_vkospi.pkl')
vix = pd.read_pickle('./working_data/df_vix.pkl')

def get_date(df, *args):
    dummy = pd.DataFrame(index = df.index.unique(), columns = ['signal'])
    dummy['signal'] = 1
    for i in args:
        dummy = dummy.multiply(i)

    res = dummy.loc[dummy['signal'] == 1].index
    return res

# 특정 요일 진입

def weekday_entry(df, weekdays = [3]):
    
    df_idx = df.index.unique()
    res = pd.DataFrame(index = df_idx, columns = ['signal'])
    res['signal'] = np.nan

    cond = df_idx.weekday.isin(weekdays)
    res.loc[cond] = 1

    return res

# 1. 과열 침체 역방향 시그널

@pd.api.extensions.register_dataframe_accessor('contra')
class contrarian:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self._df = df

    def through_bbands(self, l_or_s = 'b', length = 20, std = 2):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
        
        res = pd.DataFrame(index = self._df.index, columns = ['signal'])

        bbands = self._df.ta.bbands(length, std)
        bbands.columns = bbands.columns.str.lower()
        bbands = bbands.loc[:, (~bbands.columns.str.startswith(('bbb', 'bbp')))] # 필요없는 컬럼 삭제
        
        # 롱 시그널
        cond_long = (self._df['close'] < bbands['bbl_' + str(length) + "_" + str(float(std))]) 
        res.loc[cond_long, 'signal'] = 1
        # 숏 시그널
        cond_short = (self._df['close'] > bbands['bbu_' + str(length) + "_" + str(float(std))]) 
        res.loc[cond_short, 'signal'] = -1

        if l_or_s == 'l':
            res = res.mask(res['signal'] == -1, np.nan)
        elif l_or_s == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif l_or_s == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")

        return res
    
    def stoch_rebound(self, l_or_s ='b', k = 5, d = 3, smooth_d = 3):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
    
        res = pd.DataFrame(index = self._df.index, columns = ['signal'])
        stoch = self._df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
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
        res.loc[cond_short, 'signal'] = -1

        if l_or_s == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif l_or_s == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif l_or_s == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            


        return res

    def rsi_rebound(self, l_or_s = 'l', length = 14, scalar = 100):

        res = pd.DataFrame(index = self._df.index, columns = ['signal'])
        rsi = self._df.ta.rsi(length = length, scalar = scalar)
        
        # 롱 시그널
        cond_long_1 = rsi.shift(1) < 30 # 어제 하한선 하회
        cond_long_2 = rsi > 30 # 오늘 하한선 상회
        cond_long = cond_long_1 * cond_long_2
        res.loc[cond_long, 'signal'] = 1

        # 숏 시그널
        cond_short_1 = rsi.shift(1) > 60 # 어제 하한선 하회 # 원래 70인데 k200이 70까지 거의 안 감
        cond_short_2 = rsi < 60 # 오늘 하한선 상회
        cond_short = cond_short_1 * cond_short_2
        res.loc[cond_short, 'signal'] = -1

        if l_or_s == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif l_or_s == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif l_or_s == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            
        return res
        
    def psar_rebound(self, l_or_s = 'l'):

        psar = self._df.ta.psar()
        psar = psar.rename(columns = {'PSARl_0.02_0.2' : 'l', 'PSARs_0.02_0.2' : 's', 'PSARr_0.02_0.2' : 'signal'})    
        res = psar['signal'].mask((psar['s'].notna())&(psar['signal'] == 1), -1).to_frame()
        res = res.mask(res['signal'] == 0, np.nan)

        if l_or_s == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif l_or_s == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif l_or_s == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            
        return res

# 2. 매매 안 하는 경우 식별


class notrade:

    def no_vix_curve_invert(notrade_criteria = 0, sma_days = 20):

        df_vix = pd.read_pickle("./working_data/df_vix.pkl")
        res = pd.DataFrame(index = df_vix.index, columns = ['signal'])
        res['signal'] = 1
        #1. slope_index < 0 X
        notrade_cond = df_vix['slope_index'] < notrade_criteria
        #2. slope_index 의 sma_days 이동평균값이 하락 추세인 경우 X
        curve_score_20ma = df_vix['slope_index'].rolling(sma_days).mean()
        notrade_cond_2 = curve_score_20ma.diff(1) < 0
        res.loc[notrade_cond & notrade_cond_2, 'signal'] = np.nan

        return res

    def no_vkospi_below_n(quantile = 0.2, low_or_close = 'close', ):

        df_vkospi = pd.read_pickle("./working_data/df_vkospi.pkl")
        res = pd.DataFrame(index = df_vkospi.index, columns = ['signal'])
        res['signal'] = 1
        if low_or_close == 'low':
            limit = df_vkospi['low'].quantile(quantile)
        else:
            limit = df_vkospi['close'].quantile(quantile)
        res.loc[(df_vkospi[low_or_close] < limit), 'signal'] = np.nan

        return res
    
    def no_vkospi_above_n(high_or_close = 'close', quantile = 0.8):

        df_vkospi = pd.read_pickle("./working_data/df_vkospi.pkl")
        res = pd.DataFrame(index = df_vkospi.index, columns = ['signal'])
        res['signal'] = 1
        if high_or_close == 'high':
            limit = df_vkospi['high'].quantile(quantile)
        else:
            limit = df_vkospi['close'].quantile(quantile)
        res.loc[(df_vkospi[high_or_close] > limit), 'signal'] = np.nan

        return res
    

# 직전 고점 대비 하락폭 (통산 전고점 대비 drawdown 아님)
    



# 2. 정추세 지속 시그널

# 3. squeeze 폭발 시그널

# 4. 주가 / 지표 다이버전스

class divergence:

    def __init__(self, df):
        self.df = df
