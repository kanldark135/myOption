#%% 유틸리티 함수

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

k200 = pd.read_pickle('./working_data/df_k200.pkl')
vkospi = pd.read_pickle('./working_data/df_vkospi.pkl')
vix = pd.read_pickle('./working_data/df_vix.pkl')

def get_date_intersect(option_df, *args):
    ''' option_df : 사용하려는 옵션가격 시계열 있는 raw dataframe'''
    dummy = pd.DataFrame(index = option_df.index.unique(), columns = ['signal'])
    dummy['signal'] = 1
    for i in args:
        dummy = dummy.multiply(i)

    res = dummy.loc[dummy['signal'] == 1].index
    return res

def get_date_union(option_df, *args):
    ''' option_df : 사용하려는 옵션가격 시계열 있는 raw dataframe'''
    dummy = pd.DataFrame(index = option_df.index.unique(), columns = ['signal'])
    for i in args:
        dummy = dummy.combine_first(i.loc[dummy.index])

    res = dummy.loc[dummy['signal'] == 1].index

    return res    

#%% 

# 특정 요일 진입

def weekday_entry(option_df, weekdays = [3]):
    ''' option_df : 사용하려는 옵션가격 시계열 있는 raw dataframe'''  
    df_idx = option_df.index.unique()
    res = pd.DataFrame(index = df_idx, columns = ['signal'])
    res['signal'] = np.nan

    cond = df_idx.weekday.isin(weekdays)
    res.loc[cond] = 1

    return res


@pd.api.extensions.register_dataframe_accessor('stoch')
class stochastic:

    def __init__(self, df :[pd.Series, pd.DataFrame]):
        self.df = df

    def stoch_overtraded(self, pos = 'b', k_or_d = "k", k = 5, d = 3, smooth_d = 3):
        
        ''' 
        contrarian 계열 (pos = long이면 과매도만 return)
        default = k 기준으로 over/under 측정?
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']

        stoch['signal'] = np.nan
        stoch['signal'] = stoch['signal'].mask(stoch[k_or_d] < 20, -1) # 과매도권
        stoch['signal'] = stoch['signal'].mask(stoch[k_or_d] > 80, 1) # 과매수권

        if pos == "l": 
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan)
        elif pos == "s": 
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan) * -1
        else:
            res = stoch[['signal']]

        return res
              


# 1. 과열 침체 역방향 시그널

@pd.api.extensions.register_dataframe_accessor('contra')
class MyContrarian:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self._df = df

    def through_bbands(self, pos = 'b', length = 20, std = 2):

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

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan)
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
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
        stoch.columns = ['k', 'd']

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

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

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

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

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
    
# 2. 정추세 지속 시그널
@pd.api.extensions.register_dataframe_accessor("trend")

class MyTrend:
    def __init__(self, df):
        self.df = df

    def psar_trend(self, l_or_s = 'b', af0 = 0.02, af = 0.02, max_df = 0.2):
        
        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        psar = self.df.ta.psar(af0 = af0, af = af, max_df = max_df)
        psar = psar.rename(columns = {'PSARl_0.02_0.2' : 'l', 'PSARs_0.02_0.2' : 's', 'PSARr_0.02_0.2' : 'signal'})
        
        if l_or_s == "l":
            res = psar[['l']].mask(psar['l'].notna(), 1).rename(columns = {"l" : "signal"})
        
        elif l_or_s == "s":
            res = psar[['s']].mask(psar['s'].notna(), 1).rename(columns = {'s' : 'signal'})

        else:
            res = pd.DataFrame(data = np.where(psar['l'].notna(), 1, -1), index = psar.index, columns = ['signal'])

        return res
    
    def stoch_trend(self, l_or_s = 'b', k = 10, d = 5, smooth_d  = 5):

        '''
        실제 trend는 stoch_d 로 smoothed 된 값으로 측정
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']
        stoch = stoch - stoch.shift(1)
        stoch['signal'] = np.where(stoch['d'] >= 0, 1, -1)

        if l_or_s == "l": # 롱트렌드면 하방은 na로 처리
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan)

        elif l_or_s == "s": # 숏트렌드면 상방은 na 처리
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan) * -1
        else:
            res = stoch[['signal']]

        return res

# 3. 매매 안 하는 상황
class notrade:

    def vix_curve_invert(notrade_criteria = 0, sma_days = 20):

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

    def vkospi_below_n(quantile = 0.2, low_or_close = 'close'):

        df_vkospi = pd.read_pickle("./working_data/df_vkospi.pkl")
        res = pd.DataFrame(index = df_vkospi.index, columns = ['signal'])
        res['signal'] = 1
        if low_or_close == 'low':
            limit = df_vkospi['low'].quantile(quantile)
        else:
            limit = df_vkospi['close'].quantile(quantile)
        res.loc[(df_vkospi[low_or_close] < limit), 'signal'] = np.nan

        return res
    
    def vkospi_above_n(quantile = 0.8, high_or_close = 'close'):

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
    

# 3. 돌파매매 시그널



# 4. 주가 / 지표 다이버전스

class divergence:

    def __init__(self, df):
        self.df = df

# %%
