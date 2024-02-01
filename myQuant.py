import pandas as pd
import numpy as np


def close_only(df, column_rename = 'close'):
    try:
        df = df['Adj Close']
    except KeyError:
        try:
            potential_column_name = ['Close', "Price", "Last_Price"]
            df = df[df.columns.isin(potential_column_name)]

        except KeyError:

            if len(df.columns) == 1:
                df = df
            else:
                raise KeyError("Adj Close / Close / Last_Price 형태의 종가가 dataframe 에 존재하도록 하거나 그냥 종가 한줄 넣으세요")
    
    # 이미 DataFrame 인 경우 
    try:
        df = df.to_frame()
    except AttributeError:
        df = df
        
    df.columns = [column_rename]
    return df

def df_cumret(df, is_ret = False):
    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기'''
    if is_ret == True:
        cumret = (df + 1).cumprod() - 1
    else:
        ret = df.pct_change(1)
        ret.iloc[0] = 0
        cumret = (ret + 1).cumprod() - 1
    return cumret

def cagr(df, interval = 'D', is_ret = False):

    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기.
    interval : 'D', 'BD', 'W', 'M', 'Y' 으로 df 데이터의 interval 에 따라 정의'''

    interval_dict = {'D' : 365,
                     'BD' : 252,
                     'W' : 52,
                     'M' : 12,
                     'Y' : 1
                     }

    if is_ret == True:
        cumret = (df + 1).cumprod()
        cagr = cumret[-1] ** (interval_dict.get(interval) / len(df)) - 1
    else:
        ret = df.pct_change(1)
        ret.iloc[0] = 0
        cumret = (ret + 1).cumprod() - 1
        cagr = cumret[-1] ** (interval_dict.get(interval) / len(ret)) - 1
    return cagr

def df_drawdown(df, is_ret = False):
    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기'''
    if is_ret == True:
        cumret = (df + 1).cumprod()
        drawdown = cumret / cumret.cummax() - 1
    else:
        drawdown = df/ df.cummax() - 1
    return drawdown

def dd_from_last_top(df, is_ret = False):
    ''' drawdown 말고 최근 고점 대비 하락폭 측정'''

def mdd(df, is_ret = False):  
    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기'''
    if is_ret == True:
        cumret = (df + 1).cumprod()
        drawdown = cumret / cumret.cummax() - 1
    else:
        drawdown = df / df.cummax() - 1
    
    mdd_rolling = drawdown.cummin()
    mdd = mdd_rolling.min()
    return mdd

def annual_vol(df, interval = 'D', is_ret = False):
    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기.
    interval : 'D', 'BD', 'W', 'M', 'Y' 으로 df 데이터의 interval 에 따라 정의'''

    interval_dict = {'D' : 365,
                     'BD' : 252,
                     'W' : 52,
                     'M' : 12,
                     'Y' : 1
                     }

    if is_ret == True:
        vol = df.std() * np.sqrt(interval_dict.get(interval))
    else:
        ret = df.pct_change(1).iloc[1:]
        vol = ret.std() * np.sqrt(interval_dict.get(interval))
    return vol
    
def sharpe(df, rf = 0, is_ret = False):
    ''' df 는 종가거나 여타 가격이어야만 함 / df가 수익률인 경우 is_ret = True로 표기.
    Morningstar 방법론 적용 (CAGR 가 아니라 월 초과수익률들의 산술평균 / 월 초과수익률들의 월변동성) 으로 계산
    수익률은 전부 Monthly 로 resampling 실시

    https://awgmain.morningstar.com/webhelp/glossary_definitions/mutual_fund/mfglossary_Sharpe_Ratio.html'''
    
    rf = rf / 12
    
    if is_ret == True:
        monthly_ret = ((df + 1).resample("M").prod() - rf) -   1
        mu = monthly_ret.mean() # CAGR 가 아니라 단순 산술평균
        std = np.sqrt((monthly_ret - mu - rf).pipe(np.power, 2).sum() / (len(monthly_ret) - 1))
        sharpe = mu / std     
    else:
        ret = df.pct_change(1)
        ret.iloc[0] = 0
        monthly_ret = ((ret + 1).resample("M").prod() - rf) -   1
        mu = monthly_ret.mean() # CAGR 가 아니라 단순 산술평균
        std = np.sqrt((monthly_ret - mu - rf).pipe(np.power, 2).sum() / (len(monthly_ret) - 1))
        sharpe = mu / std 

    return sharpe

def calmar(df : pd.Series, interval = 'D', is_ret = False):
    res = cagr(df, interval = interval, is_ret = is_ret) / mdd(df, is_ret = is_ret)
    return -res

def information(df, df_bm, interval = 'D', is_ret = False):
    res = (cagr(df, interval = interval, is_ret = is_ret) - cagr(df_bm, interval = interval, is_ret = is_ret)) / annual_vol(df - df_bm, interval = interval, is_ret = is_ret)
    return res

def win_rate(df, is_ret = True):

    if is_ret == True:
        df.loc[df > 0]
        count = len(df.loc[df > 0])
    else:
        ret = df.pct_change(1)
        ret.loc[ret > 0]
        count = len(ret.loc[ret > 0])
    
    res = count / np.count_nonzero(df)
    return res

def period_return(df, interval = 'Y', is_ret = False):
    ''' Convert index to datetime / timestamp format, otherwise return error'''
    try:
        df.index = pd.to_datetime(df.index)
    except:
        raise TypeError("Index must be in time format")
    
    grouped = df.resample(interval)
    res = grouped.apply(lambda x : df_cumret(x, is_ret = is_ret).iloc[-1])
    return res

def summary(df_ret, df_bm = None, interval = 'D', rf = 0, is_ret = True):

    ''' ['ret', 'cumret', 'total_ret', 'cagr', 'annual_stdev', 'mdd', 'sharpe', 'information', 'calmar', 'win_rate] '''
    
    dummy_dict = {
        'ret' : df_ret,
        'cumret' : df_cumret(df_ret, is_ret = is_ret),
        'total_ret' : df_cumret(df_ret, is_ret = is_ret).iloc[:-1].values,
        'cagr' : cagr(df_ret, interval = interval, is_ret = is_ret),
        'annual_stdev' : annual_vol(df_ret, interval = interval, is_ret = is_ret),
        'mdd' : mdd(df_ret, is_ret = is_ret),
        'sharpe' : sharpe(df_ret, rf = 0, is_ret = is_ret),
        'information' : information(df_ret, df_bm, interval = interval, is_ret = is_ret),
        'calmar' : calmar(df_ret, interval = interval, is_ret = is_ret),
        'win_rate' : win_rate(df_ret, is_ret = is_ret)
    }

    return dummy_dict








        
