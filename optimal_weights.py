#%% 

import pandas as pd
import numpy as np
import backtest as bt
import test as tt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import differential_evolution
import os
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pathlib
import time
import test
import datetime
import ast

#%% 1) Declare environmental variables

def refine_rawdata(df_raw):
    df_raw['multiindex'] = list(zip(df_raw['strat'], df_raw['cp'], df_raw.index))
    df_raw.reset_index(inplace = True)
    df_result = df_raw.set_index('multiindex')

    return df_result

def combine_pnl(strategies, *metric_functions, start_date = '2010-01-01', commission_point = 0.002, slippage_point = 0.01):    
    db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
    # 1) 기준이 되는 df_k200
    df_k200 = bt.get_timeseries(db_path, "k200")['k200'].loc[start_date :][['close']]
    df = df_k200    
    i = 0

    all_metrics = {}
    for result in strategies:
        # result[0] : strat / result[1] : cp / result[2] : 전략분류
        pnl = test.runtest.execute(strat = result[0], cp = result[1], commission_point = commission_point, slippage_point = slippage_point)['pnl']['daily_pnl']
        pnl.name = result[0]
        
        # calculate given metrics

        single_metric = {}
        
        for function in metric_functions:
            value = function(pnl)
            single_metric[function.__name__] = value
        
        df = pd.merge(df, pnl, how = 'left', left_index = True, right_index = True)
        all_metrics[result[0]] = single_metric
        i += 1
        print(result)
    
    df_metrics = pd.DataFrame(all_metrics).T

    return df, df_metrics

def calmar(df):
    cumret = df.cumsum()
    years = (df.index[-1] - df.index[0]) / datetime.timedelta(days = 365)
    annual_gain = cumret.iloc[-1] / years
    drawdown = np.minimum(0, cumret - cumret.cummax())
    mdd = drawdown.min() 
    ratio = abs(annual_gain / mdd)

    return ratio

def sortino(df):
    daily_mean = df.mean()
    # cumret = df.cumsum()
    # years = (df.index[-1] - df.index[0]) / datetime.timedelta(days = 365)
    # annual_gain = cumret.iloc[-1] / years
    downside_std = df[df < 0].std()
    ratio = np.sqrt(252) * daily_mean / downside_std

    return ratio

def get_composite_score(df):
    # 현재는 산술평균 적용. 다만 나중에 각 metric 가지고 좀 더 유의미한 score 만드는 방법 궁리
    score = df.mean()
    return score

# (weekly 전용) 위클에서 특정일 주물의 특정일에 해당하는 counterparty 주물의 전략 반환
def get_counter_date(strat):

    split = strat.partition("_table")
    
    entry_condition = split[0]
    old_entry = entry_condition[0:3]
    split_rest = split[2].partition("_type")

    old_table = split_rest[0]
    table = ast.literal_eval(old_table)

    def temp_func_2(iterable):
        for i in iterable:
            if i.find("weekly_") == 0:
                return i
        return iterable[0]

    first_table = temp_func_2(table)

    if first_table == 'weekly_mon':
        counter = {
                'mon' : 'thu',
                'tue' : 'fri',
                'wed' : 'mon',
                'thu' : 'tue',
                'fri' : 'wed'}

    elif first_table == 'weekly_thu':
        counter = {
                'thu' : 'mon',
                'fri' : 'tue',
                'mon' : 'wed',
                'tue' : 'thu',
                'wed' : 'fri'}
        
    new_entry = counter[old_entry]

    def temp_func_2(x):
        if x == "weekly_thu":
            return "weekly_mon"
        elif x == "weekly_mon":
            return "weekly_thu"
        else:
            return "monthly"
    new_table = list(map(temp_func_2, table))

    counter_strat = strat.replace(old_entry, new_entry, 1)
    counter_strat = counter_strat.replace(old_table, str(tuple(new_table)), 1)

    return counter_strat

# 수익률 correlation 외 검토해볼만한 필터링 기준
# 1) Drawdown 기간 공유정도 측정

def get_co_drawdown(daily_pnl, threshold = -1):
    
    # 각 전략의 누적 수익률
    cum_pnl = daily_pnl.cumsum()

    # 각 전략별 drawdown 계산
    drawdown = cum_pnl - cum_pnl.cummax()

    # threshold 넘는 drawdown 발생 시 1, 아니면 0
    dd_flags = (drawdown < threshold).astype(int)

    # 전략별 동시 drawdown 발생 빈도 계산
    co_dd_matrix = dd_flags.T @ dd_flags / len(drawdown)

    return co_dd_matrix

IS_WEEKLY = False
CLUSTER_THRESHOLD = 0.2
START_DATE = '2010-01-01'
COMMISSION_POINT = 0.002
SLIPPAGE_POINT = 0.01
CLUSTERING_METRICS = [calmar]
MINIMUM_METRIC_SCORE = 0.8

if IS_WEEKLY: 
    xlsx_path = pathlib.Path.joinpath(pathlib.Path.cwd(), "전략weekly.xlsx")
else:
    xlsx_path = pathlib.Path.joinpath(pathlib.Path.cwd(), "전략monthly.xlsx")

df_raw = pd.read_excel(xlsx_path, sheet_name="1) raw", index_col=0)
df_raw = refine_rawdata(df_raw)

#%% 2) Clustering by correlation

# def cluster_by_category(df_raw, is_weekly = True, threshold = 0.3, target_metrics = [calmar], commission_point = 0.002, slippage_point = 0.01, nth_best = 3):

#     ''' target_metric : calmar / sortino + 추가 필요 '''

#     grouped = df_raw.groupby("전략분류")

#     df_result = pd.DataFrame()
#     df_pnl = pd.DataFrame()

#     start_time = time.time()

#     for name, group in grouped:
#         strategies = group.index
#         category_pnl, category_metrics = combine_pnl(strategies, *target_metrics, commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)
#         category_pnl = category_pnl.drop(columns = ['close']).fillna(0)
#         category_pnl.columns = strategies
#         category_metrics.index = strategies

#         if is_weekly == True:
#             category_pnl = category_pnl.loc['2019-09-01' : ]

#         # correlation matrix and ravelled correlation distance
#         # corr_distance considers both tendency and direction (whereas 1 - abs(corr) considers only tendency, therefore better for identifying long/short pairs)
#         corr = category_pnl.corr().fillna(0)
#         corr_distance = 1 - corr
#         dist_array = squareform(corr_distance)
        
#         # hirarchical clustering
#         linkage_matrix = linkage(dist_array, method = 'average')
#         clusters = fcluster(linkage_matrix, threshold, criterion = 'distance')

#         # 여러 metrics 동시에 반영해서 score 화 시키는 접근
#         if len(target_metrics) > 1:
# ######### composite score 만드는 방법 -> 추후 추가적으로 고민해볼 필요. 일단은 단순 산술평균 score
#             category_metrics['score'] = category_metrics.apply(get_composite_score, axis = 1)
#             score_column = 'score'
#         else:
#             category_metrics = category_metrics.rename(columns = {target_metrics[0].__name__ : 'score'})
#             score_column = 'score'

#         df_cluster = pd.DataFrame({'strategy' : category_pnl.columns, 'cluster' : clusters}).set_index('strategy')
#         df_cluster = pd.merge(df_cluster, category_metrics, left_index = True, right_index = True)

#         # selecting strategies that are #th best + return target_metric
#         df_cluster.reset_index(inplace = True)
#         result = df_cluster.groupby('cluster').apply(lambda x : x.nlargest(nth_best, columns = score_column)).reset_index(drop = True)
#         result.columns = ['strategy', 'cluster', score_column]
#         df_result = pd.concat([df_result, result], axis = 0)

#         category_pnl = category_pnl.loc[:, result['strategy']]
#         df_pnl = pd.concat([df_pnl, category_pnl], join = 'outer', axis = 1)
        
#     end_time = time.time()

#     print(f"--------------- Time taken : {(end_time - start_time) / 60} minutes")

#     df_result = df_result.set_index(['strategy'], drop = True)
#     df_pnl = df_pnl.T.drop_duplicates().T

#     return df_result, df_pnl

#1. 일단 수익률별로 1 - threshold 보다 큰 correlation 끼리 clustering
def cluster_all(df_raw, is_weekly = True, threshold = 0.3, target_metrics = [calmar], commission_point = 0.002, slippage_point = 0.01):

    ''' target_metric : calmar / sortino + 추가 필요 '''

    start_time = time.time()

    strategies = df_raw.index
    pnl, metrics = combine_pnl(strategies, *target_metrics, commission_point = commission_point, slippage_point = slippage_point)
    pnl = pnl.drop(columns = ['close']).fillna(0)
    pnl.columns = strategies
    metrics.index = strategies

    if is_weekly == True:
        pnl = pnl.loc['2019-09-01': ]

    # correlation matrix and ravelled correlation distance
    # corr_distance considers both tendency and direction (whereas 1 - abs(corr) considers only tendency, therefore better for identifying long/short pairs)
    corr = pnl.corr().fillna(0)
    corr_distance = 1 - corr
    dist_array = squareform(corr_distance)
    
    # hirarchical clustering
    linkage_matrix = linkage(dist_array, method = 'average')
    clusters = fcluster(linkage_matrix, threshold, criterion = 'distance')

    # 여러 metrics 동시에 반영해서 score 화 시키는 접근
    if len(target_metrics) > 1:
######## composite score 만드는 방법 -> 추후 추가적으로 고민해볼 필요. 일단은 단순 산술평균 score
        metrics['score'] = metrics.apply(get_composite_score, axis = 1)
        score_column = 'score'
    else:
        metrics = metrics.rename(columns = {target_metrics[0].__name__ : 'score'})
        score_column = 'score'

    df_cluster = pd.DataFrame({'strategy' : pnl.columns, 'cluster' : clusters}).set_index('strategy')
    df_cluster = pd.merge(df_cluster, metrics, left_index = True, right_index = True)

    # 일단 지금까지 clustering 된 모든 결과는 엑셀에 저장해놓기
    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as temp_writer:
        df_cluster.to_excel(temp_writer, 'df_corr_all', header = True)
        pnl.to_excel(temp_writer, "pnl_all", header = True)

    end_time = time.time()
    
    print(f"--------------- Time taken : {(end_time - start_time) / 60} minutes")
    
#2. clustering 된 성과들 중에서 가장 score 높거나 / handpicked 된 전략들 1차로 추려놓기
# 여기서 handpicked 기준은 1) 성과 / 2) 포지션 용이성 (fewer legs + 왠만하면 근월물)
def get_df_corr(manual_selection = False, nth_best = 3):

    df_cluster = pd.read_excel(xlsx_path, sheet_name = 'df_corr_all', header = 0, index_col = 0)
    pnl = pd.read_excel(xlsx_path, sheet_name = 'pnl_all', header = 0, index_col = 0)
    df_cluster.reset_index(inplace = True)
    
    if manual_selection == True:
        result = df_cluster.loc[df_cluster['manual_selection'] == 1]
        result.drop('manual_selection', axis = 1, inplace = True)
    else:
        # selecting strategies that are #th best + return target_metric
        result = df_cluster.groupby('cluster').apply(lambda x : x.nlargest(nth_best, columns = 'score')).reset_index(drop = True)
        result.drop('manual_selection', axis = 1, inplace = True)

    result.columns = ['strategy', 'cluster', 'score']

    df_result = result.set_index(['strategy'], drop = True)
    df_pnl = pnl.loc[:, df_result.index]

    return df_result, df_pnl

# 처음부터 raw 전략 다 불러와서 필터링 필요한 경우에에만 사용
# cluster_all(df_raw, is_weekly = IS_WEEKLY, threshold = CLUSTER_THRESHOLD, target_metrics = CLUSTERING_METRICS, commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)

df_auto, pnl_auto = get_df_corr(manual_selection = False, nth_best = 1)
df_manual, pnl_manual = get_df_corr(manual_selection = True)

with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
    # df_first.to_excel(writer, sheet_name = 'first', header = True)
    # pnl_first.to_excel(writer, sheet_name = 'first_pnl', header = True)
    # df_second.to_excel(writer, sheet_name = 'second', header = True)
    # pnl_second.to_excel(writer, sheet_name = 'second_pnl', header = True)

    df_auto.to_excel(writer, sheet_name = '2) df_corr_auto', header = True)
    pnl_auto.to_excel(writer, sheet_name = '2) pnl_corr_auto', header = True)
    df_manual.to_excel(writer, sheet_name = '2) df_corr_manual', header = True)
    pnl_manual.to_excel(writer, sheet_name = '2) pnl_corr_manual', header = True)

#%% 3) (위클리는 opposite pair 별도 추가) -> metric 특정값 이상만 필터링

# 기본빵은 manual 로 handpicked 우선 but auto도 마련
daily_pnl = pd.read_excel(xlsx_path, sheet_name = "2) pnl_corr_manual", header = 0, index_col = 0)
df = pd.read_excel(xlsx_path, sheet_name = "2) df_corr_manual", header = 0, index_col = 0)
# daily_pnl = pd.read_excel(xlsx_path, sheet_name = "2) pnl_corr_auto", header = 0, index_col = 0)
# df = pd.read_excel(xlsx_path, sheet_name = "2) df_corr_auto", header = 0, index_col = 0)

daily_pnl.columns = list(map(lambda x : ast.literal_eval(x), daily_pnl.columns))
df.index = list(map(lambda x : ast.literal_eval(x), df.index))

def get_opposite_strat(single_strat_combo):
    opposite_strat = get_counter_date(single_strat_combo[0])
    return (opposite_strat, single_strat_combo[1], single_strat_combo[2])

def get_weekly_opposites(df, target_metrics = [calmar]):

    df['weekly_matching'] = np.arange(len(df.index))
    strat_opposite = list(map(get_opposite_strat, df.index))
    df_opposite = df.copy()
    df_opposite.index = strat_opposite

    pnl_opposite, opposite_metrics = combine_pnl(df_opposite.index, *target_metrics, commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)
    pnl_opposite = pnl_opposite.drop(columns = ['close']).fillna(0)
    pnl_opposite = pnl_opposite.loc['2019-09-01' : ] #  어짜피 opposite 위클리밖에 없으므로...
    opposite_metrics.index = df_opposite.index
    pnl_opposite.columns = df_opposite.index

    if len(target_metrics) > 1:
######## composite score 만드는 방법 -> 추후 추가적으로 고민해볼 필요. 일단은 단순 산술평균 score
        opposite_metrics['score'] = opposite_metrics.apply(get_composite_score, axis = 1)
        score_column = 'score'
    else:
        score_column = 'score'

    df_opposite[score_column] = opposite_metrics

    return df_opposite, pnl_opposite

if IS_WEEKLY:
    df_opposite, pnl_opposite = get_weekly_opposites(df, target_metrics = CLUSTERING_METRICS)

    df = pd.concat([df, df_opposite], axis = 0)
    daily_pnl = pd.concat([daily_pnl, pnl_opposite], axis = 1)

    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        dummy_df = df.copy()
        dummy = df.reset_index()['index'].apply(pd.Series)
        dummy.index = dummy_df.index
        dummy_df[['strat', 'cp', '전략분류']] = dummy
        dummy_df.to_excel(writer, '2-2) df_weekly', header = True)
        daily_pnl.to_excel(writer, '2-2) pnl_weekly', header = True)

    # 1) score-based filtering
    df = df.loc[df['score'] > MINIMUM_METRIC_SCORE]

    # 2) filter out non-pairs IF IS_WEEKLY == TRUE; there must be at least 2 or more rows (strats) with identical weekly_matching value.
    df = df.groupby('weekly_matching').filter(lambda x : x.shape[0] > 1)
    daily_pnl = daily_pnl.loc[:, daily_pnl.columns.isin(df.index)]

    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        dummy_df = df.copy()
        dummy = df.reset_index()['index'].apply(pd.Series)
        dummy.index = dummy_df.index
        dummy_df[['strat', 'cp', '전략분류']] = dummy
        dummy_df.to_excel(writer, '3) df_metric_filtered', header = True)
        daily_pnl.to_excel(writer, '3) pnl_metric_filtered', header = True)

else:
    # 1) score-based filtering
    df = df.loc[df['score'] > 0.4]
    df['weekly_matching'] = [1000 + i for i in range(0, df.shape[0])] # weekly_matching 매칭용
    daily_pnl = daily_pnl.loc[:, daily_pnl.columns.isin(df.index)]

    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        dummy_df = df.copy()
        dummy = df.reset_index()['index'].apply(pd.Series)
        dummy.index = dummy_df.index
        dummy_df[['strat', 'cp', '전략분류']] = dummy
        dummy_df.to_excel(writer, '3) df_metric_filtered', header = True)
        daily_pnl.to_excel(writer, '3) pnl_metric_filtered', header = True)

#%% optimization    
# 위클리 + 먼슬리 통합

WEEKLY_MONTHLY_AGG = 'weekly' # weekly/monthly/agg

if WEEKLY_MONTHLY_AGG == 'weekly':

    # 일반적인 위클리 or 먼슬리로 나눠서 하는 경우
    xlsx_path = "C:/Users/kwan/Desktop/myOption/전략weekly.xlsx"
    df = pd.read_excel(xlsx_path, sheet_name = "4) final", index_col = 0, header = 0)
    df = df[df['사용'] == 1]
    daily_pnl = pd.read_excel(xlsx_path, sheet_name = "3) pnl_metric_filtered", index_col = 0, header = 0)
    try:
        daily_pnl = daily_pnl[df.index]
    except KeyError:
        existing_pnl = daily_pnl.loc[:, daily_pnl.columns.isin(df.index)]
        new_strats = list(map(ast.literal_eval, df.index[~df.index.isin(daily_pnl.columns)]))
        new_pnl, dummy = combine_pnl(new_strats, *CLUSTERING_METRICS, commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)
        new_pnl = new_pnl.drop(columns = ['close']).fillna(0)

        daily_pnl = existing_pnl.merge(new_pnl, left_index = True, right_index = True)
        daily_pnl.columns = df.index
        
    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        daily_pnl.to_excel(writer, sheet_name = "4) pnl_final", header = True)

elif WEEKLY_MONTHLY_AGG == 'monthly':

    # 일반적인 위클리 or 먼슬리로 나눠서 하는 경우
    xlsx_path = "C:/Users/kwan/Desktop/myOption/전략monthly.xlsx"
    df = pd.read_excel(xlsx_path, sheet_name = "4) final", index_col = 0, header = 0)
    df = df[df['사용'] == 1]
    daily_pnl = pd.read_excel(xlsx_path, sheet_name = "3) pnl_metric_filtered", index_col = 0, header = 0)
    try:
        daily_pnl = daily_pnl[df.index]
    except KeyError:
        existing_pnl = daily_pnl.loc[:, daily_pnl.columns.isin(df.index)]
        new_strats = list(map(ast.literal_eval, df.index[~df.index.isin(daily_pnl.columns)]))
        new_pnl, dummy = combine_pnl(new_strats, *CLUSTERING_METRICS, commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)
        new_pnl = new_pnl.drop(columns = ['close']).fillna(0)
        daily_pnl = existing_pnl.merge(new_pnl, left_index = True, right_index = True)
        daily_pnl.columns = df.index
        
    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        daily_pnl.to_excel(writer, sheet_name = "4) pnl_final", header = True)

elif WEEKLY_MONTHLY_AGG == 'agg':

    xlsx_path = "C:/Users/kwan/Desktop/myOption/전략agg.xlsx"
    df = pd.read_excel(xlsx_path, sheet_name = "4) final", index_col = 0, header = 0)
    df = df[df['사용'] == 1]
    strats = list(map(ast.literal_eval, df.index))
    daily_pnl, dummy = combine_pnl(strats, *CLUSTERING_METRICS, start_date = '2019-09-01', commission_point = COMMISSION_POINT, slippage_point = SLIPPAGE_POINT)
    daily_pnl = daily_pnl.drop(columns = ['close']).fillna(0)
    daily_pnl.columns = df.index

    with pd.ExcelWriter(xlsx_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
        daily_pnl.to_excel(writer, sheet_name = "4) pnl_final", header = True)

else:
    raise ValueError("WEEKLY_MONTHLY_AGG MUST BE ONE OF weekly/monthly/agg")

#%% 

def get_metrics(weight, daily_pnl, if_return = False, initial_capital = 400):

    ''' initial_capital = 400 = 1억원'''
    
    # Convert units to integer weights
    # weights = np.round(units / np.sum(np.array(units)) * 100).astype(int)
    weights = np.round(weight).astype(int)
    weights_series = pd.Series(weights, index=daily_pnl.columns)


    # 가중치 적용된 일일 손익 및 누적 손익 계산
    if if_return == False:
        # 일일수익금
        weighted_daily_pnl = (daily_pnl * weights_series).sum(axis=1)
        # 누적수익금
        cum_pnl = weighted_daily_pnl.cumsum()
        # MDD 계산
        mdd = (cum_pnl - cum_pnl.cummax()).min()
        # 연간 수익 계산
        annualize_factor = (weighted_daily_pnl.index[-1] - weighted_daily_pnl.index[0]).days / 365
        annual_gain = cum_pnl.iloc[-1] / annualize_factor
    else:
        # 일일수익률
        weighted_daily_pnl = (daily_pnl * weights_series).sum(axis=1) / initial_capital
        # 누적수익률
        cum_pnl = (1 + weighted_daily_pnl).cumprod() - 1
        # MDD 계산
        mdd = ((cum_pnl + 1) / (cum_pnl + 1).cummax() - 1).min()
        # 연간 수익 계산
        annualize_factor = (weighted_daily_pnl.index[-1] - weighted_daily_pnl.index[0]).days / 365
        annual_gain = np.power(cum_pnl.iloc[-1] + 1, 1/annualize_factor) -1    

    # 평균 일일 수익률
    daily_mean = weighted_daily_pnl.mean()

    # 일일 수익률의 표준편차
    daily_std = weighted_daily_pnl.std()
    
    # 하방 변동성 (음수 수익률만 고려)
    downside_std = weighted_daily_pnl[weighted_daily_pnl < 0].std()
    
    # Calmar Ratio
    calmar_ratio = annual_gain / abs(mdd)
    
    # Sharpe Ratio (무위험 수익률 = 0 가정)
    sharpe_ratio = np.sqrt(252) * daily_mean / daily_std 
    
    # Sortino Ratio
    sortino_ratio = np.sqrt(252) * daily_mean / downside_std
    
    return {
        'calmar': calmar_ratio,
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'volatility': daily_std,
        'mdd': mdd,
        'annual_gain': annual_gain,
        'weights': weights  # Return the integer weights for reference
    }

def optimize_portfolio(daily_pnl, df, objective='calmar', min_weight=0, max_weight=10, if_return=False, initial_capital=400):

    # 전략을 weekly_matching 기준으로 그룹화
    matching_groups = df.groupby("weekly_matching").groups
    group_keys = list(matching_groups.keys())

    # 각 전략을 해당 그룹 인덱스에 매핑 (→ 동일 그룹이면 동일 인덱스를 참조)
    group_index_map = {}
    for group_idx, key in enumerate(group_keys):
        for strat in matching_groups[key]:
            group_index_map[strat] = group_idx

    # 최적화 대상 변수 수 = unique group 수
    n_groups = len(group_keys)
    bounds = [(min_weight, max_weight)] * n_groups

    pnl = daily_pnl.fillna(0).copy()

    # 목표 함수 정의
    def fitness(group_weights):

        group_weights = np.round(group_weights).astype(int)

        # 각 전략에 그룹 weight를 일괄 적용 (→ hard constraint 방식)
        strat_weights = np.array([group_weights[group_index_map[strat]] for strat in pnl.columns])

        metrics = get_metrics(strat_weights, daily_pnl = pnl, if_return = if_return, initial_capital = initial_capital)

        if objective == 'calmar':
            base_value = -metrics['calmar']  # 최소화 문제로 변환
        elif objective == 'sharpe':
            base_value = -metrics['sharpe']
        elif objective == 'sortino':
            base_value = -metrics['sortino']
        elif objective == 'min_vol':
            base_value = metrics['volatility']
        else:
            raise ValueError("objective must be one of 'calmar', 'sharpe', or 'sortino'")
        
        return base_value
    
    # 여러 번의 최적화 실행 (local minima/maxima 문제 해결)
    best_result = None
    best_value = float('inf')
    result_vector = []
    
    for _ in range(20):  # 20회에 걸쳐 서로 다른 random starting point 에서 시작해서 제일 좋은거 추리기
        start_time = time.time()
        result = differential_evolution(
            fitness,
            bounds=bounds,
            popsize=50,  # 20에서 50으로 증가
            mutation=(0.5, 1.0),
            recombination=0.7,
            maxiter=100,  # 50에서 100으로 증가
            seed=np.random.randint(0, 10000),  # 랜덤 seed 사용
            workers=1  # 병렬 처리 활성화
        )

        result_vector.append(result.x)        
        # if result.fun < best_value:
        #     best_value = result.fun
        #     best_result = result
        end_time = time.time()

        print(f"time taken for a single loop : {(end_time - start_time) / 60} minutes")
    weights = []
    for x in result_vector:
        optimal_weight = np.round(x).astype(int)
        strat_weights = np.array([optimal_weight[group_index_map[strat]] for strat in daily_pnl.columns])
        weights.append(strat_weights)

    return weights, daily_pnl.columns

# 다양한 목적함수로 최적화 실행
objectives = ['calmar']
# objectives = ['calmar', 'sharpe', 'sortino']
results = {}

for obj in objectives:

    IF_RETURN = False

    print(f"\nOptimizing for {obj}...")
    weights, strategies = optimize_portfolio(daily_pnl, df, obj, min_weight = 0, max_weight = 10, if_return = IF_RETURN, initial_capital = 400)
    metric_list = []
    for weight in weights:
        metric = get_metrics(weight, daily_pnl, if_return = IF_RETURN, initial_capital = 400)
        metric_list.append(metric)
    
    weights = pd.DataFrame(weights, columns = strategies)
    metrics = pd.DataFrame(metric_list)

    
    print(f"\nOptimal Strategy Units ({obj}):")
    for strat, unit in zip(strategies, weights):
        print(f"{strat}: {unit} units")
    
    # 결과 저장
    results[obj] = {
        'units': weights,
        'strategies': strategies,
        'metric': metrics
    }

with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    for obj in results:
        weights = results[obj]['units']
        strategies = results[obj]['strategies']
        metrics_df = results[obj]['metric']
        metrics_df.to_excel(writer, sheet_name=f'metrics_{obj}')

        df = df.merge(weights.T, left_index=True, right_index=True)
        df.to_excel(writer, sheet_name=f'result_{obj}')

#%% 시각화
plt.figure(figsize=(12, 6))
for obj in objectives:
    weights = results[obj]['units']
    strategies = results[obj]['strategies']
    weighted_pnl = (daily_pnl[strategies] * pd.Series(weights, index=strategies)).sum(axis=1)
    cum_pnl = weighted_pnl.cumsum()
    cum_pnl.plot(label=obj)

plt.title('Cumulative PnL by Objective')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
