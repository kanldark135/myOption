#%% 
import pandas as pd
import numpy as np
import compute as compute
from option_calc import get_option_data, get_calendar, get_skewness

# %% preprocessing for analytics

def to_percentile(df):
    res = df['atm'].rank(pct = True)
    df['percentile'] = res
    return df

def generate_group_quantile(df, by, q = 5):
    bins = pd.qcut(df[by], q = q)
    grouped = df.groupby(bins)
    return grouped

# %% IV analytics

# 원하는 atm_value 구간설정
front_atm = None
back_atm = [0.16, 0.17]

# 원하는 dte_value 구간설정
front_dte = [14, 28]
back_dte = [35, 70]

# 원하는 groupby 기준
_groupby = 'atm'  # 'atm', 'close', 'dte' 정도 가능

cf = get_option_data('iv', 'monthly', cycle = 'front', callput = 'call', moneyness_ub = 40, atm_range= front_atm, dte_range = front_dte)
cb = get_option_data('iv', 'monthly', cycle = 'back', callput = 'call', moneyness_ub = 40, atm_range = back_atm, dte_range = back_dte)
pf = get_option_data('iv', 'monthly', cycle = 'front', callput = 'put', moneyness_ub = 40, atm_range = front_atm, dte_range = front_dte)
pb = get_option_data('iv', 'monthly', cycle = 'back', callput = 'put', moneyness_ub = 40, atm_range = back_atm, dte_range = back_dte)

#1. IV 별로 각 quantile 내 IV의 그룹
pb_grouped = generate_group_quantile(pb, by = _groupby, q = 5)
cb_grouped = generate_group_quantile(cb, by = _groupby, q = 5)

#2. 조건별로 각 quantile 내 skew의 그룹
pb_skew = get_skewness(pb)
cb_skew = get_skewness(cb)
pb_skew_grouped = generate_group_quantile(pb_skew, by = _groupby, q = 5)
cb_skew_grouped = generate_group_quantile(cb_skew, by = _groupby, q = 5)

#3. 조건별로 각 quantile 내 skew의 그룹
pb_calendar = get_calendar(pf, pb)
cb_calendar = get_calendar(cf, cb)
pb_calendar_grouped = generate_group_quantile(pb_calendar, by = _groupby, q = 5)
cb_calendar_grouped = generate_group_quantile(cb_calendar, by = _groupby, q = 5)

