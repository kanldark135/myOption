#%% 유틸리티 함수

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

k200 = pd.read_pickle('./working_data/df_k200.pkl')
vkospi = pd.read_pickle('./working_data/df_vkospi.pkl')
vix = pd.read_pickle('./working_data/df_vix.pkl')
df_monthly = pd.read_pickle("./working_data/df_monthly.pkl")
df_weekly = pd.read_pickle("./working_data/df_weekly.pkl")

def get_date_intersect(option_df, *args): # df_1과 df_2가 동시에 적용디는 날짜만 분리
    ''' option_df : 사용하려는 옵션가격 시계열 있는 raw dataframe'''
    dummy = pd.DataFrame(index = option_df.index.unique(), columns = ['signal'])
    dummy['signal'] = 1
    for i in args:
        dummy = dummy.multiply(i)

    res = dummy.loc[dummy['signal'] == 1].index
    return res

def get_date_union(option_df, *args): # df_1 또는 df_2 둘중 어느 날이나 포함
    ''' option_df : 사용하려는 옵션가격 시계열 있는 raw dataframe'''
    dummy = pd.DataFrame(index = option_df.index.unique(), columns = ['signal'])
    for i in args:
        dummy = dummy.combine_first(i.loc[dummy.index])

    res = dummy.loc[dummy['signal'] == 1].index

    return res  

#  df_1 날짜중에 df_2 날짜는 빼고 나머지에 진입 : 별도 함수 대신 get_date_intersect(df_1, flip(df_2)) 로  df_1과 df_2 의 여집합


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
class stoch_signal:

    def __init__(self, df :[pd.Series, pd.DataFrame]):
        self.df = df

    def is_overtraded(self, pos = 'b', k_or_d = "k", k = 5, d = 3, smooth_d = 3):
        
        '''
        building block 1) 
        과열 / 침체 여부 (20이하/ 80이상)
        default = k 기준으로 over/under 측정?
        과매도권  = 'l'
        과매수권 = 's'
        both = 'b'
        '''
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']
        stoch['signal'] = np.nan

        stoch['signal'] = stoch['signal'].mask(stoch[k_or_d] < 20, -1) # 과매도권
        stoch['signal'] = stoch['signal'].mask(stoch[k_or_d] > 80, 1) # 과매수권

        if pos == "l": 
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan) * -1
        elif pos == "s": 
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan)
        else:
            res = stoch[['signal']]
        return res

    def is_updown(self, pos = 'b', k_or_d = 'k', k = 5, d = 3, smooth_d = 3):

        ''' buidling block 2)
        전일대비 상승중인지 하락중인지 여부
        상승만 = 'l'
        하락만 = 's'
        both = 'b'
        '''
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']
        stoch['signal'] = np.nan
        
        stoch['signal'].loc[stoch[k_or_d] >= stoch[k_or_d].shift(1)] = 1
        stoch['signal'].loc[stoch[k_or_d] < stoch[k_or_d].shift(1)] = -1

        if pos == "l": 
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan)
        elif pos == "s": 
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan) * -1
        else:
            res = stoch[['signal']]

        return res        

    def rebound1(self, pos ='b', k = 5, d = 3, smooth_d = 3):

        '''
        반전신호 : 전날까지 과매도 / 과매수권에 있다가 + 당일 K가 D를 반대로 돌파하는
        long_only = 상승반전만
        short_only = 하락반전만
        both = 둘다
        '''
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']
        stoch['signal'] = np.nan

        # 롱 시그널
        cond_long_1 = stoch['k'].shift(1) <= 20 # K가 전날 20 밑에 (오늘은 상관 없음)
        cond_long_2 = stoch['k'] > stoch['d'] # K가 오늘 D를 상향돌파
        cond_long = cond_long_1 * cond_long_2
        stoch.loc[cond_long, 'signal'] = 1

        # 숏 시그널
        cond_short_1 = stoch['k'].shift(1) > 80 # K가 전날 80 위에 (오늘은 상관 없음)
        cond_short_2 = stoch['k'] < stoch['d'] # K가 오늘 D를 하향돌파
        cond_short = cond_short_1 * cond_short_2
        stoch.loc[cond_short, 'signal'] = -1

        if pos == "l": 
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan)
        elif pos == "s": 
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan) * -1
        else:
            res = stoch[['signal']]      

        return res

    def rebound2(self, pos ='b', k_or_d = 'k', k = 5, d = 3, smooth_d = 3):

        '''
        반전신호 : 전날까지 과매도 / 과매수권에 있다가 + 당일 K가 D를 반대로 돌파하는
        long_only = 상승반전만
        short_only = 하락반전만
        both = 둘다
        '''
        stoch = self.df.ta.stoch(k = k, d = d, smooth_d = smooth_d)
        stoch.columns = ['k', 'd']
        stoch['signal'] = np.nan
        
        # 롱 시그널
        cond_long_1 = stoch[k_or_d].shift(1) <= 20 # K가 전날 20 밑에 
        cond_long_2 = stoch[k_or_d].shift(1) > stoch[k_or_d] # K가 전날대비 상승
        cond_long = cond_long_1 * cond_long_2
        stoch.loc[cond_long, 'signal'] = 1

        # 숏 시그널
        cond_short1 = stoch[k_or_d].shift(1) >= 80 # K가 전날 80 위에
        cond_short2 = stoch[k_or_d].shift(1) < stoch[k_or_d] # K가 전날대비 하락
        cond_short = cond_short1 * cond_short2
        stoch.loc[cond_short, 'signal'] = -1

        if pos == "l": 
            res = stoch[['signal']].mask(stoch['signal'] == -1, np.nan)
        elif pos == "s": 
            res = stoch[['signal']].mask(stoch['signal'] == 1, np.nan) * -1
        else:
            res = stoch[['signal']]      

        return res         

@pd.api.extensions.register_dataframe_accessor('bbands')
class bband_signal:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self.df = df

    def outside_bbands(self, pos = 'b', length = 20, std = 2):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
        res = pd.DataFrame(index = self.df.index, columns = ['signal'])

        bbands = self.df.ta.bbands(length, std)
        bbands.columns = ['low', 'mid', 'high', 'width', 'prob']
        bbands['signal'] = np.nan

        # 롱 시그널
        cond_long = self.df['close'] < bbands['low']
        bbands.loc[cond_long, 'signal'] = 1
        # 숏 시그널
        cond_short = self.df['close'] > bbands['high']
        bbands.loc[cond_short, 'signal'] = -1

        res = bbands[['signal']]

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan)
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")

        return res

    def through_bbands(self, pos = 'b', length = 20, std = 2):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
        res = pd.DataFrame(index = self.df.index, columns = ['signal'])

        bbands = self.df.ta.bbands(length, std)
        bbands.columns = ['low', 'mid', 'high', 'width', 'prob']
        bbands['signal'] = np.nan

        # 볼밴 하방돌파 시점 -> 롱
        cond_long = (self.df['close'] < bbands['low']) & (self.df['close'].shift(1) > bbands['low'].shift(1))
        bbands.loc[cond_long, 'signal'] = 1
        # 볼밴 상방돌파 시점 -> 숏
        cond_short = (self.df['close'] > bbands['high']) & (self.df['close'].shift(1) < bbands['high'])
        bbands.loc[cond_short, 'signal'] = -1

        res = bbands[['signal']]

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan)
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")

        return res
@pd.api.extensions.register_dataframe_accessor('rsi')
class rsi_signal:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self.df = df


    def rebound(self, pos = 'l', length = 14, scalar = 100):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        res = pd.DataFrame(index = self.df.index, columns = ['signal'])
        rsi = self.df.ta.rsi(length = length, scalar = scalar)
        
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

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            
        return res
    
@pd.api.extensions.register_dataframe_accessor('psar')
class psar_signal:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self.df = df

    def rebound(self, pos = 'l'):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        psar = self.df.ta.psar()
        psar = psar.rename(columns = {'PSARl_0.02_0.2' : 'l', 'PSARs_0.02_0.2' : 's', 'PSARr_0.02_0.2' : 'signal'})    
        res = psar['signal'].mask((psar['s'].notna())&(psar['signal'] == 1), -1).to_frame()
        res = res.mask(res['signal'] == 0, np.nan)

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            
        
        return res

    def trend(self, pos = 'b', af0 = 0.02, af = 0.02, max_df = 0.2):
        
        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        psar = self.df.ta.psar(af0 = af0, af = af, max_df = max_df)
        psar = psar.rename(columns = {'PSARl_0.02_0.2' : 'l', 'PSARs_0.02_0.2' : 's', 'PSARr_0.02_0.2' : 'signal'})
        
        if pos == "l":
            res = psar[['l']].mask(psar['l'].notna(), 1).rename(columns = {"l" : "signal"})
        
        elif pos == "s":
            res = psar[['s']].mask(psar['s'].notna(), 1).rename(columns = {'s' : 'signal'})

        elif pos =='b':
            res = pd.DataFrame(data = np.where(psar['l'].notna(), 1, -1), index = psar.index, columns = ['signal'])

        else:
            raise ValueError('pos must be l / s / b')
        
        return res
    
@pd.api.extensions.register_dataframe_accessor('supertrend')
class supertrend_signal:

    def __init__(self, df:[pd.DataFrame, pd.Series]):
        self.df = df

    def rebound(self, pos = 'l' , length = 7, atr_multiplier = 3):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''
        
        supertrend = self.df.ta.supertrend(length = length, multiplier = atr_multiplier)
        supertrend.columns = ['trend', 'signal', 'long', 'short'] 
        bool = supertrend['signal'] != supertrend['signal'].shift(1)
        res = supertrend[['signal']].where(bool, np.nan)

        if pos == 'l':
            res = res.mask(res['signal'] == -1, np.nan) 
        elif pos == 's':
            res = res.mask(res['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수로 전환
        elif pos == 'b':
            res = res
        else:
            raise ValueError("l_or_s must be l, s or b")            
        return res
    
    def trend(self, pos = 'b', length = 7, atr_multiplier = 3):

        '''
        long_only = 'l'
        short_only = 's'
        both = 'b'
        '''

        supertrend = self.df.ta.supertrend(length = length, multiplier = atr_multiplier)
        supertrend.columns = ['trend', 'signal', 'long', 'short']

        if pos == 'l':
            res = supertrend[['signal']].mask(supertrend['signal'] == -1, np.nan)
        elif pos == 's':
            res = supertrend[['signal']].mask(supertrend['signal'] == 1, np.nan) * -1 # 숏만 필터링할거면 양수
        elif pos == 'b':
            res = supertrend[['signal']]
        else:
            raise ValueError('pos must be l / s / b')

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
    
# 저가매수 전략 : 직전 고점 대비 (전고점 말고)

def change_recent(df, change : float, ohlc = 'close'):

    # 추가 보완 (or 안해도 될...) : 저가는 전고점 / 현재 low랑 비교, 고가는 전저점 / 현재 high 랑 비교
    # but 장중매매 입장에서 현재 가격이 당일 저점인지 고점인지 어짜피 모르므로 (백테스팅도 안 됨) 종가-종가 비교가 현실적이라는 판단

    res = pd.DataFrame(index = df.index, columns = ['signal'])

    if ohlc in ['close', 'open', 'high', 'low']:
        df = df[ohlc]
    else:
        raise IndexError("ohlc must be close / open / high / low")

    if change <= 0: # 전고점대비 change 만큼 낮으면 long signal
        
        higher_than_yesterday = df >= df.shift(1)
        df_higher = df.where(higher_than_yesterday).fillna(method = 'ffill')

        df = pd.concat([df, df_higher], axis = 1, join = 'inner')
        df.columns = [ohlc, f"{ohlc}_high"]
        df['chg'] = df[ohlc] / df[f"{ohlc}_high"] - 1

        res = res.mask(df['chg'] <= change, 1)
    
    if change > 0: # 전저점대비 change 만큼 높으면 short signal
            
        lower_than_yesterday = df < df.shift(1)
        df_lower = df.where(lower_than_yesterday).fillna(method = 'ffill')

        df = pd.concat([df, df_lower], axis = 1, join = 'inner')
        df.columns = [ohlc, f"{ohlc}_lower"]
        df['chg'] = df[ohlc] / df[f"{ohlc}_lower"] - 1

        res = res.mask(df['chg'] > change, -1)

    return res

# 3. 돌파매매 시그널



# 4. 주가 / 지표 다이버전스

class divergence:

    def __init__(self, df):
        self.df = df

# %%
