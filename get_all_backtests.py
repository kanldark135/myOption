#%% setting up
import pandas as pd
import numpy as np
import get_entry_exit
import datetime
import time
import joblib
import matplotlib.pyplot as plt
import backtest as bt
import pathlib
import itertools
import typing
import duckdb

db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")
df_k200 = bt.get_timeseries(db_path, "k200")['k200']

class metrics:

    def __new__(cls, df_result):
        instance = super().__new__(cls)
        instance.df = df_result
        res = instance.calculate_metrics()
        return res
        
    def calculate_metrics(self):

        basic_metrics = self.basic(self.df)
        sortino_metrics = self.sortino(self.df)
        calmar_metrics = self.calmar(self.df)
        avg_profit_loss_metrics = self.avg_profit_loss(self.df)
        win_count_metrics = self.win_count(self.df)
        
        # 모든 메트릭 결과를 병합
        all_metrics = {**basic_metrics, **sortino_metrics, **calmar_metrics, 
                       **avg_profit_loss_metrics, **win_count_metrics}
        
        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame([all_metrics])
        
        return result_df

    def basic(self, df):

        pnl = df['pnl']

        res_dict = dict(
        cum_ret = pnl['cum_pnl'].iloc[-1],
        max_ret = pnl['cum_pnl'].max(),
        mdd = pnl['dd'].min()
        )

        return res_dict
        
    def sortino(self, df):
        pnl = df['pnl']
        years = (pnl['cum_pnl'].index[-1] - pnl['cum_pnl'].index[1]) / datetime.timedelta(days = 365)
        annual_gain = pnl['cum_pnl'].iloc[-1] / years
        downside_std = np.sqrt(252) * pnl['daily_pnl'].loc[pnl['daily_pnl'] < 0].std()

        ratio = annual_gain / downside_std
        return dict(sortino = ratio)

    def calmar(self, df):
        '''
        최소 1이상은 나와줘야
        '''

        pnl = df['pnl']
        years = (pnl['cum_pnl'].index[-1] - pnl['cum_pnl'].index[1]) / datetime.timedelta(days = 365)
        annual_gain = pnl['cum_pnl'].iloc[-1] / years
        mdd = pnl['dd'].min()

        ratio = annual_gain / abs(mdd)
        return dict(calmar = ratio)

    def avg_profit_loss(self, df):
        '''
        1 이상이면 좋은거
        '''
        res = df['res']
        avg_profit = res['cum_pnl'].loc[res['cum_pnl'] > 0].mean()
        avg_loss = res['cum_pnl'].loc[res['cum_pnl'] < 0].mean()

        ratio = abs(avg_profit / avg_loss)
        return dict(avg_pnl_ratio = ratio)
    
    def win_count(self, df):
        res = df['res'].value_counts('whystop')
        
        res_dict = dict(
        dte = res.get('dte', 0),
        win = res.get('win', 0),
        loss = res.get('loss', 0),
        stop = res.get('stop', 0)
        )

        return res_dict

def combine_pnl(*pnls):
    
    i = 0
    for pnl in pnls:

        if i == 0:
            df = pnl['pnl']
            i += 1
        else:
            df = df + pnl['pnl']
            i += 1

    df['dd'] = df['cum_pnl'] - df['cum_pnl'].cummax()

    print(f"final return : {df['cum_pnl'].iloc[-1]}")
    print(f"max return : {df['cum_pnl'].max()}")
    print(f"mdd : {df['dd'].min()}")

    return df

def generate_iterables(*args):
    container = []
    for item in args:
        if not isinstance(item, (list, tuple, dict, np.ndarray)):
            container.append([item])
        else:
            container.append(item)
            
    res = itertools.product(*container)
    return res

def generate_moneyness_offset(number_offset_legs = 3, interval_between_legs = 2.5, min_offset = 2.5, max_offset = 7.5):

    ''' number_offset_legs = 0 -> reference value로만 진행되는 single leg 전략'''

    list_append = []
    multiplier = list(itertools.product(range(int(min_offset / 2.5), int(max_offset / 2.5) + 1), repeat = number_offset_legs))
    for i in multiplier:

        if all(i[j] < i[j + 1] for j in range(len(i) - 1)): # 직전 offset보다 큰 (크거나 작은도 X) 값만 의미있음
            
            list_append.append(list(map(lambda x : x * interval_between_legs, i)))        

    return list_append

def generate_delta_offset(number_offset_legs, divisor, reference_delta, max_diff = 0.25):

    ''' reference delta list 에서 간편하게 divisor 만큼 나눈 값을 offset으로 보고 거기에 대해서 [a>b] 적용한 pair'''

    divided_list = []
    current_list = reference_delta.copy()

    for i in range(number_offset_legs + 1):
        divided_list.append(current_list)
        current_list = list(map(lambda x : np.round(x / divisor, 2), current_list))

    list_append = []

    multiplier = list(itertools.product(*divided_list))
    for i in multiplier:

        cond1 = all(np.abs(i[j]) > np.abs(i[j + 1]) for j in range(len(i) - 1)) # 1. 직전 offset보다 큰 (크거나 작은도 X) 값만 의미있음
        cond2 = all(np.abs(i[j]) - np.abs(i[j+1]) < max_diff for j in range(len(i)- 1)) # 직전 offset과 max_diff 보다는 덜 차이나야 함 (그 이상 차이나면 그냥 네이키드랑 다를 바 없어)

        if all([cond1, cond2]): 
            
            list_append.append(list(i))        

    return list_append

def generate_query_vars(entry_dates,
                        table,
                        types : list,
                        term,
                        ref_values,
                        offset_values,
                        offset_type : list,
                        is_callput = False
                        ):

    ''' reference + offset 조합
    1) moneyness + moneyness offset : 쿼리로 구현. 제일 간단
    2) delta + moneyness offset : 쿼리로 구현. ref_term + ref_delta 의 행사가 찾고 -> 그 행사가 + moneyness offset 만큼의 행사가 찾아서 -> offset_term 에 해당하는 월물 선택
    3) pct + moneyness offset : 쿼리로 구현.
    3) (X) moneyness + delta offset : 이렇게 할 이유 없음 X
    4) delta + delta offset : 별도로 구현
    '''

    query_vars = dict()

    if not isinstance(types, list):
        types = [types]
        
    if is_callput == False:

    # 1. offset_type 없는 경우 : (단일행사가 또는 지정행사가 leg 들의 조합)
        if len(offset_type) == 0:
            for type in types:
                ref_value = ref_values[type]
                query_vars[type] = list(generate_iterables(entry_dates, table, type, term, ref_value))

    # 2. offset 있는 경우
        else:
            if not isinstance(offset_type, list):
                raise TypeError("offset type 은 무조건 list 구조여야함")

            for offset in offset_type:
                ref = offset.split("&")[0]
                off = offset.split("&")[1]

                if ref == 'moneyness' and off == "moneyness":
                    ref_value = ref_values[ref]
                    offset_value = offset_values[off]
                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_value, offset_value))
                    
                elif ref == 'pct' and off == "moneyness":
                    ref_value = ref_values[ref]
                    offset_value = offset_values[off]
                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_value, offset_value))

                elif ref == 'delta' and off == "moneyness":
                    ref_value = ref_values[ref]
                    offset_value = offset_values[off]
                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_value, offset_value))

                elif ref == 'delta' and off == "delta":
                    ref_value = offset_values[ref]
                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_value))

                else:
                    raise ValueError("offset type 은 list 내에 'moneyness_moneyness', 'delta_moneyness', 'pct_moneyness', 'delta_delta' 이렇게 4개 값만 가져야 함")
                
    elif is_callput == True:

    # 1. offset_type 없는 경우 : (단일행사가 또는 지정행사가 leg 들의 조합)
        if len(offset_type) == 0:
            for type in types:

                if type == 'moneyness':
                    ref_value = ref_values[type]
                    opposite_ref = ref_value
                    pair_value = list(itertools.product(ref_value, opposite_ref))

                else:
                    ref_value = ref_values[type]
                    opposite_ref = list(lambda x: [-1 * x[0]] if len(x) == 1 else -1 * x, ref_value)
                    pair_value = list(itertools.product(ref_value, opposite_ref))

                query_vars[type] = list(generate_iterables(entry_dates, table, type, term, pair_value))

    # 2. offset 있는 경우
        else:
            if not isinstance(offset_type, list):
                raise TypeError("offset type 은 무조건 list 구조여야함")

            for offset in offset_type:
                ref = offset.split("&")[0]
                off = offset.split("&")[1]

                if ref == 'moneyness' and off == "moneyness":
                    ref_value = ref_values[ref]
                    opposite_ref = ref_value
                    offset_value = offset_values[off]

                    ref_pair = list(itertools.product(ref_value, offset_value))
                    opposite_pair = list(itertools.product(opposite_ref, offset_value))

                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_pair, opposite_pair))
                    
                elif ref == 'pct' and off == "moneyness":
                    ref_value = ref_values[ref]
                    opposite_ref = list(map(lambda x : -1 * x, ref_value))
                    offset_value = offset_values[off]

                    ref_pair = list(itertools.product(ref_value, offset_value))
                    opposite_pair = list(itertools.product(opposite_ref, offset_value))

                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_pair, opposite_pair))
                    
                elif ref == 'delta' and off == "moneyness":
                    ref_value = ref_values[ref]
                    opposite_ref = list(map(lambda x : -1 * x, ref_value))
                    offset_value = offset_values[off]

                    ref_pair = list(itertools.product(ref_value, offset_value))
                    opposite_pair = list(itertools.product(opposite_ref, offset_value))

                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_pair, opposite_pair))
                    
                elif ref == 'delta' and off == "delta":
                    ref_value = offset_values[ref]
                    opposite_ref = (-1 * np.array(ref_value)).tolist()
                    query_vars[offset] = list(generate_iterables(entry_dates, table, ref, term, ref_value, opposite_ref))

                else:
                    raise ValueError("offset type 은 list 내에 'moneyness_moneyness', 'delta_moneyness', 'pct_moneyness', 'delta_delta' 이렇게 4개 값만 가져야 함")
                         
    return query_vars

def add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars, apply_costs, commission_point, slippage_point):
    
    df = pd.DataFrame()

    stop_dates = stop_dates
    dte_stop = dte_stop
    profit = profit
    loss = loss
    is_complex_pnl = is_complex_pnl
    is_intraday_stop = is_intraday_stop
    start_date = start_date
    end_date = end_date
    show_chart = show_chart
    use_polars = use_polars
    apply_costs = apply_costs
    commission_point = commission_point
    slippage_point = slippage_point

    trade_vars = list(generate_iterables(stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop))

    for stop_key, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop in trade_vars:

        name = strat_name + f'exit{stop_key}_dte{dte_stop}_p{profit}_l{loss}_{is_complex_pnl}'

        result = queried.equal_inout(stop_dates[stop_key],
                                    dte_stop,
                                    profit,
                                    loss,
                                    is_complex_pnl,
                                    is_intraday_stop,
                                    start_date,
                                    end_date,
                                    show_chart,
                                    use_polars,
                                    apply_costs,
                                    commission_point,
                                    slippage_point)
        
        if result == None:
            continue

        else:
            res = metrics(result)
            res.index = [name]

            df = pd.concat([df,res], axis = 0)
        
    return df

#%%  entry 조건

if __name__ == "__main__":
    
    ''' 아래는 query_vars : entry_dates / table / (ref) types / term / ref_values / leg_numbers / offset_values --------------------------------------------------------------- '''

    start_date = '2010-01-01'
    df_result = pd.DataFrame()
    placeholder = []
    placeholder2 = []

    cp = "P" # C/P/B
    
    apply_costs = True # 수수료와 슬리피지 적용 여부
    commission_point = 0.002 # 수수료
    slippage_point = 0.01 # 슬리피지

    # 차이점
    # tables = [('weekly_mon',), ('weekly_thu',)] # leg 한개인 naked 매매를 각각 월요일물 / 목요일물에 대해 두번 loop
    # tables = [('weekly_mon', 'weekly_thu')] # leg 두개인 전략의 각 leg에 대해 leg1 = 월요일물 / leg2 = 목요일물

    tables = [('weekly_mon', 'weekly_mon'), ('weekly_thu', 'weekly_thu')] # 디폴트 값. leg당 하나씩 적용하는
    # tables = [('weekly_mon'), ('monthly', 'monthly'), ('monthly', 'monthly'), ('monthly', 'monthly')] # leg당 여러개 적용하는데 내가 원하는것만 적용하는 경우
    # tables = list(generate_iterables(['weekly_mon', 'monthly'])) # leg 당 모든 조합의 가짓수 (복원추출)

    terms = [(1, 1), (1, 1)] # 디폴트 참조. 2leg 전략에서 앞에꺼는 근월물 / 뒤에까는 차월물 적용
    # terms = [(1, 1, 1)] * 8 # 위클리의 모든 조합 적용
    # term = [(1,), (2,)] # single leg 전략에서 각각 근월물 / 차월물로 loop 두번 돌리는 형태로 적용 (X2)
    # term = list(generate_iterables([1, 2], [1, 2])) # leg 당 모든 조합의 가짓수 (복원추출)

    # volumes = {'leg' : [(1, -1), (1, -2), (2, -1), (2, -3), (3, -1)]}
    # volumes = {'leg' : [(-1, 1), (-1, 2), (-2, 1), (-2, 3), (-3, 1)]} # 디폴트 값
    volumes = {'leg' : [(1, -1)]} # 한 사이드에 여러 조합 적용할때
    # volumes = {'leg' : [(-1,)]} # 한 사이드에 한 leg만
    # volumes = {'leg' : [(1, -1)], 'opposite' : [(1, -1)]} # cp = 'B' 일때. leg 당 opposite 이 반드시 있어야 하고, 순서대로임

    iterable = list(zip(tables, terms))

    for table, term in iterable:

        print("Now prcoessing :", table, term)
    
        types = ['delta', 'moneyness']

        first_table = table[0]
        first_term = term[0]
        try:
            second_table = table[1]
            second_term = term[1]
        except:
            pass

        entry_dates = dict(
            mon = df_k200.weekday(0),
            tue = df_k200.weekday(1),
            wed = df_k200.weekday(2),
            thu = df_k200.weekday(3),
            fri = df_k200.weekday(4),

            monivbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', option_path)),
            monivbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', option_path)),
            tueivbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', option_path)),
            tueivbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', option_path)),
            wedivbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', option_path)),
            wedivbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', option_path)),
            thuivbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', option_path)),
            thuivbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', option_path)),
            friivbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.25, 'lower', option_path)),
            friivbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'lower', option_path)),

            monivabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', option_path)),
            monivabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', option_path)),
            tueivabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', option_path)),
            tueivabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', option_path)),
            wedivabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', option_path)),
            wedivabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', option_path)),
            thuivabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', option_path)),
            thuivabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', option_path)),
            friivabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.5, 'upper', option_path)),
            friivabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.iv_filter(cp, first_table, first_term, 0.75, 'upper', option_path)),
            
            monskewbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', option_path)),
            monskewbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', option_path)),
            tueskewbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', option_path)),
            tueskewbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', option_path)),
            wedskewbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', option_path)),
            wedskewbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', option_path)),
            thuskewbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', option_path)),
            thuskewbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', option_path)),
            friskewbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.skew_filter(first_table, first_term, 0.25, 'lower', option_path)),
            friskewbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'lower', option_path)),

            monskewabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', option_path)),
            monskewabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', option_path)),
            tueskewabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', option_path)),
            tueskewabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', option_path)),
            wedskewabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', option_path)),
            wedskewabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', option_path)),
            thuskewabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', option_path)),
            thuskewabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', option_path)),
            friskewabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.skew_filter(first_table, first_term, 0.5, 'upper', option_path)),
            friskewabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.skew_filter(first_table, first_term, 0.75, 'upper', option_path)),

            # moncalendarabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', option_path)),
            # moncalendarabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', option_path)),
            # tuecalendarabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', option_path)),
            # tuecalendarabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', option_path)),
            # wedcalendarabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', option_path)),
            # wedcalendarabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', option_path)),
            # thucalendarabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', option_path)),
            # thucalendarabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', option_path)),
            # fricalendarabove50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'upper', option_path)),
            # fricalendarabove75 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.75, 'upper', option_path))

            # moncalendarbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', option_path)),
            # moncalendarbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(0), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', option_path))
            # tuecalendarbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', option_path)),
            # tuecalendarbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(1), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', option_path)),
            # wedcalendarbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', option_path)),
            # wedcalendarbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(2), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', option_path)),
            # thucalendarbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', option_path)),
            # thucalendarbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(3), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', option_path)),
            # fricalendarbelow50 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.5, 'lower', option_path)),
            # fricalendarbelow25 = get_entry_exit.get_date_intersect(df_k200.weekday(4), get_entry_exit.iv.calendar_filter(cp, first_table, first_term, second_table, second_term, 0.25, 'lower', option_path))
                                                                                                                                                                                                                                                                                                                                                                    
            # psar_trend_up = get_entry_exit.get_date_intersect(df_k200.weekday(0), df_k200.psar.trend('l')), # psar 상승하고있을때 콜 진입
            # stoch_turn_up = df_k200.stoch.rebound1('l', 10, 3, 3), # psar 상승하고있을때 콜 진입
            # rsi_turn_up = df_k200.rsi.rebound('l', 14, low = 30, high = 60), # psar 상승하고있을때 콜 진입
            # price_rebound = df_k200.priceaction.change_recent(change = -0.03), # 직전 상승 대비 -3%(전고점 아님) 떨어지면 진입
            # price_rebound2 = df_k200.priceaction.change_recent(change = -0.05) # 직전 상승 대비 -3%(전고점 아님) 떨어지면 진입
            
            psar_trend_down = get_entry_exit.get_date_intersect(df_k200.weekday(0), df_k200.psar.trend('s')), # psar 상승하고있을때 콜 진입 
            stoch_turn_down = df_k200.stoch.rebound1('s', 10, 3, 3), # psar 상승하고있을때 콜 진입
            rsi_turn_down = df_k200.rsi.rebound('s', 14, low = 30, high = 60), # psar 상승하고있을때 콜 진입
            price_drop = df_k200.priceaction.change_recent(change = 0.03), # 직전 상승 대비 -3%(전고점 아님) 떨어지면 진입
            price_drop2 = df_k200.priceaction.change_recent(change = 0.05) # 직전 상승 대비 -3%(전고점 아님) 떨어지면 진입
        )

        ref_values = dict(
            delta = [0.05, 0.1, 0.2, 0.3, 0.4] if cp in ['C', 'B']\
                else [-0.05, -0.1, -0.2, -0.3, -0.4],
            # moneyness = [10, 15, 20, 25, 30, 35, 40, 45, 50] if first_table == 'monthly'\
            moneyness = [10, 20, 30, 40, 50] if first_table == 'monthly'\
                else [2.5, 5, 7.5, 10, 12.5]
            # pct = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        )

        leg_numbers = 1
        offset_values = dict(
            delta = generate_delta_offset(number_offset_legs = leg_numbers, divisor = 1.5, reference_delta = ref_values['delta'], max_diff = 0.25),
            moneyness = generate_moneyness_offset(number_offset_legs = leg_numbers, interval_between_legs=2.5, min_offset=2.5, max_offset = 7.5)
            # 캘린더 전용
            # delta = list(generate_iterables(ref_values['delta'], ref_values['delta'])), # 캘린더로 할때 델타 offset 말고 아예 동일한 델타까지 열어두기
            # moneyness = generate_moneyness_offset(number_offset_legs = leg_numbers, interval_between_legs=2.5, min_offset=0, max_offset = 15) # 캘린더로 할때 offset 말고 근월물 등가격까지 차월물도 열어두기
        )

        table = [table]
        term = [term]

        query_vars = generate_query_vars(entry_dates,
                                        table,
                                        types,
                                        term,
                                        ref_values,
                                        offset_values,
                                        offset_type = ['moneyness&moneyness', 'delta&moneyness', 'delta&delta']) # offset_type = 'moneyness&moneyness', 'delta&moneyness', 'pct&moneyness', 'delta&delta'
        
        placeholder.append(query_vars)

        query_vars_both = generate_query_vars(entry_dates,
                                        table,
                                        types,
                                        term,
                                        ref_values,
                                        offset_values,
                                        offset_type = ['moneyness&moneyness', 'delta&moneyness', 'delta&delta'],
                                        is_callput = True) # offset_type = 'moneyness&moneyness', 'delta&moneyness', 'pct&moneyness', 'delta&delta'
        
        placeholder2.append(query_vars_both)
        
        '''아래서부터는 trade_vars : stop_dates / dte_stop / profift / loss / is_complex_pnl / is_intrade_stop / start_date / end_date  ----------------------------------------'''

        stop_dates = {
            'noexit' : []
        }

        dte_stop = [1]

        # p/l 
        profit = [0.5, 1, 1.5, 2]
        loss = [-0.5, -0.75]

        is_complex_pnl = False
        is_intraday_stop = False
        start_date = start_date
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
        show_chart = False
        use_polars = True

        ''' ----------------------------------------------------------------------------------------------------   '''

        start_time = time.time()

    # 콜 / 풋 단방향인 경우
        if cp in ['C', 'P']:

            # 전략부분 쿼리조건
            for strike_type in query_vars.keys():

            # moneyness 기반 offset 적용하는 경우
                if strike_type in ['moneyness*5moneyness', 'delta&moneyness', 'pct&moneyness']:
                    for entry, table, type, term, ref_value, offset_value in query_vars[strike_type]:
                        
                        entry_date = entry_dates.get(entry)
                        
                        for volume in volumes['leg']:

                            leg1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : cp,
                                        'type' : type,
                                        'select_value' : ref_value,
                                        'term' : term[0],
                                        'volume' : volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            leg2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : cp,
                                        'type' : type,
                                        'select_value' : ref_value,
                                        'term' : term[0],
                                        'volume' : volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999],
                                        'offset' : offset_value[0],
                                        'term_offset' : term[1]
                                        }
                            # leg3 = {'entry_dates' : entry_date,
                            #             'table' : table[2],
                            #             'cp' : cp,
                            #             'type' : type,
                            #             'select_value' : ref_value,
                            #             'term' : term[0],
                            #             'volume' : volume[2],
                            #             'dte' : [1, 999],
                            #             'iv_range' : [0, 999],
                            #             'offset' : offset_value[1],
                            #             'term_offset' : term[2]
                            #             }

                            queried = bt.backtest(leg1, leg2, option_path=option_path)
                            strat_name = f"{entry}_table{table}_type{strike_type}_term{term}_volume{volume}_ref{ref_value}_ofst{offset_value}_"
                            res = add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars, apply_costs, commission_point, slippage_point)
                            df_result = pd.concat([df_result, res], axis = 0)

            # 델타기반 offset 적용하는 경우 -> 사실상 offset 이 아닌 따로 도출된 델타값으로 불러오는셈 but term 은 다르게
                elif strike_type in ['delta&delta']:
                    for entry, table, type, term, ref_value in query_vars[strike_type]:

                        entry_date = entry_dates.get(entry)

                        for volume in volumes['leg']:

                            leg1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : cp,
                                        'type' : type,
                                        'select_value' : ref_value[0],
                                        'term' : term[0],
                                        'volume' : volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            leg2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : cp,
                                        'type' : type,
                                        'select_value' : ref_value[1],
                                        'term' : term[1],
                                        'volume' : volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            # leg3 = {'entry_dates' : entry_date,
                            #             'table' : table[2],
                            #             'cp' : cp,
                            #             'type' : type,
                            #             'select_value' : ref_value[2],
                            #             'term' : term[2],
                            #             'volume' : volume[2],
                            #             'dte' : [1, 999],
                            #             'iv_range' : [0, 999],
                            #             }

                            queried = bt.backtest(leg1, leg2, option_path=option_path)
                            strat_name = f"{entry}_table{table}_type{strike_type}_term{term}_volume{volume}_ref{ref_value}_"
                            res = add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars, apply_costs, commission_point, slippage_point)
                            df_result = pd.concat([df_result, res], axis = 0)

        #    offset 없이 임의로 행사가 지정하는 경우
        #         else:
        #             for entry, table, type, term, ref_value in query_vars[strike_type]:

        #                 entry_date = entry_dates.get(entry)
                
        #                 for volume in volumes['leg']:

        #                     call1 = {'entry_dates' : entry_date,
        #                                 'table' : table,
        #                                 'cp' : cp,
        #                                 'type' : type,
        #                                 'select_value' : ref_value,
        #                                 'term' : term[0],
        #                                 'volume' : volume[0],
        #                                 'dte' : [1, 999],
        #                                 'iv_range' : [0, 999]
        #                                 }
        #                     call2 = {'entry_dates' : entry_date,
        #                                 'table' : table,
        #                                 'cp' : cp,
        #                                 'type' : type,
        #                                 'select_value' : ref_value[1],
        #                                 'term' : term[1],
        #                                 'volume' : volume[1],
        #                                 'dte' : [1, 999],
        #                                 'iv_range' : [0, 999]
        #                                 }
        #                     call3 = {'entry_dates' : entry_date,
        #                                 'table' : table,
        #                                 'cp' : cp,
        #                                 'type' : type,
        #                                 'select_value' : ref_value[2],
        #                                 'term' : term[2],
        #                                 'volume' : volume[2],
        #                                 'dte' : [1, 999],
        #                                 'iv_range' : [0, 999]
        #                                 }

        #                 queried = bt.backtest(call1, option_path = option_path)
        #                 strat_name = f"{entry}_{table}_{type}_term{term}_ref{ref_value}_volume{volume}_"
        #                 res = add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars)
        #                 df_result = pd.concat([df_result, res], axis = 0)

        #     전술변수 : 손절, 익절, 청산조건 + 시작일 등

    # 양매도 양매수만
        elif cp == 'B':

            start_time = time.time()

            # 전략부분 쿼리조건
            for strike_type in query_vars_both.keys():

            # moneyness 기반 offset 적용하는 경우
                if strike_type in ['moneyness&moneyness', 'delta&moneyness', 'pct&moneyness']:
                    for entry, table, type, term, ref_value, opposite_value in query_vars_both[strike_type]:

                        entry_date = entry_dates.get(entry)
                        
                        ref = ref_value[0]
                        ref_offsets = ref_value[1]
                        opposite_ref = opposite_value[0]
                        opposite_ref_offsets = opposite_value[1]
            
                        j = 0

                        for volume in volumes['leg']:
                            opposite_volume = volumes['opposite'][j]
                            j += 1

                            call1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : 'C',
                                        'type' : type,
                                        'select_value' : ref,
                                        'term' : term[0],
                                        'volume' : volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            call2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : 'C',
                                        'type' : type,
                                        'select_value' : ref,
                                        'term' : term[0],
                                        'volume' : volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999],
                                        'offset' : ref_offsets[0],
                                        'term_offset' : term[1]
                                        }
                            put1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : 'P',
                                        'type' : type,
                                        'select_value' : opposite_ref,
                                        'term' : term[0],
                                        'volume' : opposite_volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            put2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : 'P',
                                        'type' : type,
                                        'select_value' : opposite_ref,
                                        'term' : term[0],
                                        'volume' : opposite_volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999],
                                        'offset' : opposite_ref_offsets[0],
                                        'term_offset' : term[1]
                                        }

                            queried = bt.backtest(call1, call2, put1, put2, option_path=option_path)
                            strat_name = f"{entry}_table{table}_type{strike_type}_term{term}_callvolume{volume}_putvolume{opposite_volume}_callref{ref}_callofst{ref_offsets}_putref{opposite_ref}_putofst{opposite_ref_offsets}_"
                            res = add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars, apply_costs, commission_point, slippage_point)
                            df_result = pd.concat([df_result, res], axis = 0)

            # 델타기반 offset 적용하는 경우 -> 사실상 offset 이 아닌 따로 도출된 델타값으로 불러오는셈 but term 은 다르게
                elif strike_type in ['delta&delta']:
                    for entry, table, type, term, ref_value, opposite_value in query_vars_both[strike_type]:

                        entry_date = entry_dates.get(entry)

                        j = 0

                        for volume in volumes['leg']:
                            opposite_volume = volumes['opposite'][j]
                            j += 1

                            call1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : 'C',
                                        'type' : type,
                                        'select_value' : ref_value[0],
                                        'term' : term[0],
                                        'volume' : volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            call2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : 'C',
                                        'type' : type,
                                        'select_value' : ref_value[1],
                                        'term' : term[1],
                                        'volume' : volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            put1 = {'entry_dates' : entry_date,
                                        'table' : table[0],
                                        'cp' : 'P',
                                        'type' : type,
                                        'select_value' : opposite_value[0],
                                        'term' : term[0],
                                        'volume' : opposite_volume[0],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }
                            put2 = {'entry_dates' : entry_date,
                                        'table' : table[1],
                                        'cp' : 'P',
                                        'type' : type,
                                        'select_value' : opposite_value[1],
                                        'term' : term[1],
                                        'volume' : opposite_volume[1],
                                        'dte' : [1, 999],
                                        'iv_range' : [0, 999]
                                        }

                            queried = bt.backtest(call1, call2, put1, put2, option_path=option_path)
                            strat_name = f"{entry}_table{table}_type{strike_type}_term{term}_callvolume{volume}_putvolume{opposite_volume}_callref{ref_value}_putref{opposite_value}_"
                            res = add_trade_result(queried, strat_name, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop, start_date, end_date, show_chart, use_polars, apply_costs, commission_point, slippage_point)
                            df_result = pd.concat([df_result, res], axis = 0)

    end_time = time.time()
    print(f'{end_time - start_time} has taken to complete loop')

    df_result = df_result.sort_values('calmar', ascending = False)
