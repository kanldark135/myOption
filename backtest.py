import pandas as pd
import numpy as np
import sqlite3
import get_entry_exit
import datetime
import time
import typing
import joblib
import matplotlib.pyplot as plt
import inspect
import pathlib
import polars as pl


# 2. entry 및 해당 옵션의 만기까지의 전체 로우 뽑아내는 함수
# > get_option(한개 entry_date) 과 같아서 불필요함
def get_single_month(entry_date,
                    table : typing.Literal['monthly', 'weekly_thu', 'weekly_mon'],
                    cp : typing.Literal['C', 'P'],
                    type : typing.Literal['strike', 'moneyness', 'delta', 'pct'],
                    select_value : int | float,
                    term : typing.Literal[1, 2, 3, 4, 5],
                    dte : list = [0, 999]):
    
    ''' 
    TERM은 0 : 상장된 종목중 가장 최근월물을 의미
    
    1) 일단 DTE 적용해서 최소한 이정도 DTE 이상 / 이하 종목 선택 후
    2) 그 다음 TERM 적용해서 가장 최근월물 / 차근월물... 선택

    예) DTE 0, 56 / TERM = 0
    > DTE 0~56까지 남은 종목 중 가장 최근월물
    '''

    conn = sqlite3.connect(option_path)

    #1. 개별 string으로 입력하거나 아님 dataframe 의 index를 받는 경우로 상정
    
    entry_date = pd.to_datetime(entry_date).date().strftime('%Y-%m-%d')

    #2. 에러발생시 raise ValueError
    if table not in ['monthly', 'weekly_thu', 'weekly_mon']:
        raise ValueError("invalid table name")

    if cp not in ['C', 'P']:
        raise ValueError("C or P")

    if type not in ['strike', 'moneyness', 'delta', 'pct']:
        raise ValueError("select_strike must be strike, moneyness, delta or pct")

    if not isinstance(select_value, (int, float)):
        raise ValueError("select_value must be int or float")
    
    #3.

    if type == "strike":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (ORDER BY dte ASC) AS term
            FROM {table}
            WHERE date = '{entry_date}'
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND {type} = {select_value}
                ORDER BY dte ASC
        ),
        code AS (
        SELECT code
            FROM temp_data
            where term = {term}
        )
        SELECT m.*  
            FROM {table} m
            INNER JOIN code ON m.code = code.code
            WHERE m.date >= '{entry_date}'
        '''

    elif type == "moneyness":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (ORDER BY dte ASC) AS term
            FROM {table}
            WHERE date = '{entry_date}'
            AND cp = '{cp}'
            AND dte between {dte[0]} and {dte[1]}
            AND {type} = {select_value}
                ORDER BY dte ASC
        ),
        code AS (
        SELECT code
            FROM temp_data
            where term = {term}
        )
        SELECT m.*  
            FROM {table} m
            INNER JOIN code ON m.code = code.code
            WHERE m.date >= '{entry_date}'
            '''

    elif type == "delta":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            ABS(delta - ({select_value})) AS delta_difference,
            DENSE_RANK() OVER (ORDER BY dte ASC) AS term
            FROM {table}
            WHERE date = '{entry_date}'
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND {type} BETWEEN {select_value - 0.1} AND {select_value + 0.1}
                ORDER BY delta_difference ASC
        ),
        code AS (
        SELECT code
            FROM temp_data
            WHERE term = {term}
            LIMIT 1 OFFSET 0
        )
        SELECT m.*  
            FROM {table} m
            INNER JOIN code ON m.code = code.code
            WHERE m.date >= '{entry_date}';
        '''
    
    elif type == "pct":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (ORDER BY dte ASC) AS term,
            ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
            FROM {table}
            WHERE date = '{entry_date}'
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND strike BETWEEN close_k200 * {1 + select_value} - 1.25 AND close_k200 * {1 + select_value} + 1.25
                ORDER BY strike_difference ASC
        ),
        code AS (
        SELECT code
            FROM temp_data
            WHERE term = {term}
            LIMIT 1 OFFSET 0;
        )
        SELECT m.*
            FROM {table} m
            INNER JOIN code ON m.code = code.code
            WHERE m.date <= '{entry_date}';
        '''

    df = pd.read_sql(query, conn, index_col = 'date')
    conn.close()

    return df
    
#3. 각 date 에 대해 해당 옵션의 만기까지의 전체 로우를 뽑아내되, 이걸 여러 entry dates 에 대해서 한번에 일괄 적용
# 하는 이유 : 파이썬 내에서 각 date 에 대해서 for date in entry_dates 연산이 너무 느림
# (np.vectorize / df['date'].apply 등 모두 생각했던것보다 느림)
# 일단 일괄적으로 불러다가 df_aggregate.groupby 실시 후 각 subgroup 에 대해서 손절/익절 연산 실시

def get_option(entry_dates,
                    table : typing.Literal['monthly', 'weekly_thu', 'weekly_mon'],
                    cp : typing.Literal['C', 'P'],
                    type : typing.Literal['strike', 'moneyness', 'delta', 'pct'],
                    select_value : int | float,
                    term : typing.Literal[1, 2, 3, 4, 5],
                    dte : list = [1, 999],
                    iv_range : list = [0, 999],
                    offset = 0,
                    *args, # 주문수량 등등 받기 위한 역할 없는 dummy arguments
                    **kwargs
                    ) : # dte = 0으로 두면 만기날 종가에 포지션 들어가는것처럼 되면서 익/손 0짜리 dummy 포지션 생성
    
    ''' 
    TERM : 상장된 종목중 가장 최근월물 순으로 1, 2, 3, 4, 5
    
    구조는
    1) 일단 DTE 적용해서 최소한 이정도 DTE 이상 / 이하 종목 선택 후
    2) 그 다음 TERM 적용해서 "해당 DTE 내에서 최근월물 / 차근월물... 선택"

    예 1) DTE [0, 56] / Term = 1
    > DTE 0~56까지 남은 종목 중 가장 최근월물
    예 2) DTE [24, 70] / Term = 1
    > DTE 24~70 사이에서 가장 최근월물 (찐 최근월물은 DTE = 3 이렇게 남아있다고 하더라도 해당사항 아님)

    만약 그냥 찐 최근월물만 매매하고 싶으면 그냥 dte = [0, 999] 로 두고 term = 1 하면 됨
    '''

    # 변경 필요
    option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_option.db")
    conn = sqlite3.connect(option_path)

    #1. 인덱스의 자료형 상관없이 전부 str 형태로 변형
    entry_dates = pd.to_datetime(entry_dates, format = 'mixed').strftime("%Y-%m-%d")

    if isinstance(entry_dates, str):
        formatted_dates = entry_dates # 날짜 하나밖에 없으면 있는 그대로 사용
    else:
        entry_dates = list(entry_dates)
        formatted_dates = "', '".join(entry_dates)

    #2. 에러발생시 raise ValueError
    if table not in ['monthly', 'weekly_thu', 'weekly_mon']:
        raise ValueError("invalid table name")

    if cp not in ['C', 'P']:
        raise ValueError("C or P")

    if type not in ['strike', 'moneyness', 'delta', 'pct']:
        raise ValueError("select_strike must be strike, moneyness, delta or pct")

    if not isinstance(select_value, (int, float)):
        raise ValueError("select_value must be int or float")
    
    #3.
    if offset == 0:
        if type == "strike":
            query = f'''
            WITH term_selected_data AS (
            SELECT * 
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} and {dte[1]}
                    AND {type} = {select_value}
                    )
                WHERE term = {term}
            ),
            iv_selected_data AS (
                SELECT t.*
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            )
            SELECT m.*, i.term, i.date as entry_date 
                FROM {table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date;
            '''

        elif type == "moneyness":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} and {dte[1]}
                    AND {type} = {select_value}
                    )
                WHERE term = {term}
            ),
            iv_selected_data AS (
                SELECT t.*
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            )
            SELECT m.*, i.term, i.date as entry_date 
                FROM {table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date;
            '''

        elif type == "delta": # delta between 0.0 and 0.5; 왜냐면 dte = 1~2 + 주가 저 바닥에 200포인트 아래있을때는 2.5pt 차이로 델타가 매우벌어짐
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    ABS(delta - ({select_value})) AS delta_difference,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} AND {dte[1]}
                    AND {type} BETWEEN {select_value - 0.3} AND {select_value + 0.3}
                )
                WHERE term = {term}
            ),
            iv_selected_data AS (
            SELECT t.*,
                ROW_NUMBER() OVER (PARTITION BY t.date ORDER BY t.delta_difference ASC) AS equal_deltadiff_cols
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            ),
            closest_delta_data AS (
            SELECT i.*
                FROM iv_selected_data i
                WHERE i.equal_deltadiff_cols = 1
            )
            SELECT m.*, d.term, d.date as entry_date
                FROM {table} m
                INNER JOIN closest_delta_data d ON m.code = d.code
                WHERE m.date >= d.date;
            '''
        
        elif type == "pct":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
                    ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} AND {dte[1]}
                    AND strike BETWEEN close_k200 * {1 + select_value} - 1.25 AND close_k200 * {1 + select_value} + 1.25
                )
            WHERE term = {term}
            ),
            iv_selected_data AS (
            SELECT t.*,
                ROW_NUMBER() OVER (PARTITION BY t.date ORDER BY t.strike_difference ASC) AS equal_strikediff_cols            
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]} 
            ),
            closest_pct_data AS (
            SELECT i.*
                FROM iv_selected_data i
                WHERE i.equal_strikediff_cols = 1
            )
            SELECT m.*, p.term, p.date as entry_date
                FROM {table} m
                INNER JOIN closest_pct_data p ON m.code = p.code
                WHERE m.date >= p.date;
            '''

    else: # offset 있는 경우 실질적으로 point offset밖에 안 쓸거 같아 이것만 적용
        if type == "strike":
            query = f'''
            WITH term_selected_data AS (
            SELECT * 
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} and {dte[1]}
                    AND {type} = {select_value} + {offset}
                    )
                WHERE term = {term}
            ),
            iv_selected_data AS (
                SELECT t.*
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            )
            SELECT m.*, i.term, i.date as entry_date 
                FROM {table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date;
            '''

        elif type == "moneyness":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} and {dte[1]}
                    AND {type} = {select_value} + {offset}
                    )
                WHERE term = {term}
            ),
            iv_selected_data AS (
                SELECT t.*
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            )
            SELECT m.*, i.term, i.date as entry_date 
                FROM {table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date;
            '''

        elif type == "delta": # delta between 0.0 and 0.5; 왜냐면 dte = 1~2 + 주가 저 바닥에 200포인트 아래있을때는 2.5pt 차이로 델타가 매우벌어짐
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, cp, exp, strike, iv,
                    ABS(delta - ({select_value})) AS delta_difference,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} AND {dte[1]}
                    AND {type} BETWEEN {select_value - 0.3} AND {select_value + 0.3}
                )
                WHERE term = {term}
            ),
            iv_selected_data AS (
            SELECT t.*,
                ROW_NUMBER() OVER(PARTITION BY t.date ORDER BY t.delta_difference ASC) AS equal_deltadiff_cols
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            ),
            closest_delta_data AS (
            SELECT i.*
                FROM iv_selected_data i
                WHERE i.equal_deltadiff_cols = 1
            ),
            strike_adjusted_data AS (
                SELECT d.*, d.strike + {offset} as new_strike
                FROM closest_delta_data d
            )
            SELECT m.*, k.term, k.date as entry_date
                FROM {table} m
                INNER JOIN strike_adjusted_data k
                ON m.cp = k.cp
                AND m.exp = k.exp
                AND m.strike = k.new_strike
                WHERE m.date >= k.date;
            '''
        
        elif type == "pct":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, cp, exp, strike,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
                    ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
                    FROM {table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {dte[0]} AND {dte[1]}
                    AND strike BETWEEN close_k200 * {1 + select_value} - 1.25 AND close_k200 * {1 + select_value} + 1.25
                )
                WHERE term = {term}
            ),
            iv_selected_data AS (
            SELECT t.*,
                ROW_NUMBER() OVER (PARTITION BY t.date ORDER BY t.strike_difference() ASC) AS equal_strikediff_cols
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            ),
            closest_pct_data AS (
            SELECT t.*
                FROM iv_selected_data i
                WHERE i.equal_strikediff_cols = 1
            ),
            strike_adjusted_data AS (
            SELECT d.*, d.strike + {offset} as new_strike
                FROM closest_pct_data d
            )
            SELECT m.*, k.term, k.date as entry_date
                FROM {table} m
                INNER JOIN strike_adjusted_data k
                ON m.cp = k.cp
                AND m.exp = k.exp
                AND m.strike = k.new_strike
                WHERE m.date >= k.date;
            '''

    df = pd.read_sql(query, conn, index_col = 'date')
    conn.close()

    return df

# 나중에 

# 멀티leg 고려사항
# ㅁ 진입일이 같고 exit 도 한번에 하는 가장 일반적인 경우
# index랑 entry_date 으로 join 하면 될 것으로 추정
# 한가지 고려할게 먼슬리랑 위클리랑 캘린더 할 수도 있음 -> 일단 차이 없어보임

# ㅁ 과거데이터에 데이터 없는 특정 레그 (가령 275 282.5인데 282.5 없는경우)
# 빼버려야한다고 봄...


class backtest:

    def __init__(self, *args : dict):

        ''' dte 와 iv 는 optional'''

        start_time = time.time()

        self.required_keys = {'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'volume'}
        self.raw_df, self.order_volume = self.fetch_and_process(*args)
        self.concat_df = self.join_legs_on_entry_date(self.raw_df)
        self.aux_data = get_timeseries(pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db"), 'k200', 'vkospi')
        self.k200 = self.aux_data['k200']
        self.vkospi = self.aux_data['vkospi']

        end_time = time.time()
        print(f"importing data : {end_time - start_time} seconds")

    def fetch_and_process(self, *args : dict):
        
        dict_raw_df = {}
        dict_order_volume = {}

        def generate_leg_name(arg, default_params):

            iv_range = arg.get('iv_range', default_params['iv_range'].default)
            dte = arg.get('dte', default_params['dte'].default)
            offset = arg.get('offset', default_params['offset'].default)
            return f"{arg['table']}_{arg['cp']}_{arg['type']}_{arg['select_value']}_{arg['term']}_{dte}_{iv_range}_{offset}_{arg['volume']}"
        
        def process_arg(arg, default_params):

            if not isinstance(arg, dict):
                raise TypeError(f"arg는 최소 'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'volume' 를 키값으로 가지는 dict 여야 함")
            if not self.required_keys.issubset(arg.keys()):
                raise ValueError(f"Missing 최소 required keys : {self.required_keys}")
            
            leg_name = generate_leg_name(arg, default_params)

            df = get_option(**arg)
            return leg_name, df, arg['volume']
        
        default_params = inspect.signature(get_option).parameters

        if len(args) > 5:

            results = joblib.Parallel(n_jobs = -1)(
                joblib.delayed(process_arg)(arg, default_params) for arg in args
                )
        else:
            results = [process_arg(arg, default_params) for arg in args]            
        
        for leg_name, df, volume in results:
            dict_raw_df[leg_name] = df
            dict_order_volume[leg_name] = volume

        return dict_raw_df, dict_order_volume

    def join_legs_on_entry_date(self, raw_df): # 멀티leg 들 죄다 date/entry_date 두개 기준으로 join 해놓는 모듈러 함수

        concat_df = None
        for key in raw_df.keys():
            df = raw_df[key].copy()
            volume = self.order_volume[key]

            # 1. 각종 작업
                        
            df = df.assign(
                value=df['adj_price'] * volume,
                delta=df['delta'] * volume,
                gamma=df['gamma'] * volume,
                theta=df['theta'] * volume,
                vega=df['vega'] * volume
            )
            
            df.index = pd.to_datetime(df.index)
            df[['exp_date', 'entry_date']] = df[['exp_date', 'entry_date']].apply(pd.to_datetime, format = '%Y-%m-%d')
            df = df.set_index([df.index, 'entry_date']) # 인덱스를 (date, entry_date) 으로 멀티인덱스화
            df.columns = pd.MultiIndex.from_product([[key], df.columns]) # 컬럼도 leg별 분류를 위해서 멀티컬럼화
            
            if concat_df is None:
                concat_df = df
                
            else:
                concat_df = pd.merge(concat_df, df, 
                                        how = 'inner', 
                                        # 'outer' : concat_df / df 에서 없는 로우(= 해당시점에 존재하지 않던 행사가) 도 빈칸으로 들어감
                                        # 'inner' : 해당시점에 df 들중에 행사가 존재하지 않는 leg 있으면 그 기간은 전부 날려버림
                                        left_index = True,
                                        right_index = True
                                        )
            
        return concat_df
    
        # 각 leg 들을 날짜랑 / entry_date 두개 기준으로 outer join
        # date 는 어떤 종목 언제 들어가도 다 동일한 가운데,
        # leg 들의 entry 가 서로 같다면, inner join 이나 outer join 이나 결과 같음 > entry가 같은 조건은 여기서 데이터 합산 완료
        # leg 들의 entry 가 서로 다른 경우, 일단 outer join 으로 df_concat 에 전부 포함시켜 놓고 나중에 따로 처리

    def equal_inout(self,
                    stop_dates = [],
                    dte_stop = 0,
                    profit = 0,
                    loss = 0,
                    is_complex_pnl = False,
                    is_intraday_stop = False,
                    start_date = None,
                    end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
                    show_chart = False,
                    use_polars = False
                    ):

        start_time = time.time()

        # 1. 추출된 데이터들 전부 시간형으로 변경        
        # start_date / end_date 만들어놓은 이유 : 나중에 가격데이터 일부만 빼서 train / test set 나눌려고
        df = self.concat_df.copy()

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        stop_dates = pd.to_datetime(stop_dates)

        if not df.empty:

            df = df.loc[(slice(start_date, end_date))]
            df_daterange = df.index.get_level_values(level = 'date')
            date_range = self.k200.loc[slice(df_daterange[0], df_daterange[-1])].index  # get_option 은 실제 entry_date 에 해당하는 날만 가져오는 반면, 분석은 해당 기간 전체애 대해 하기 위해서 별도의 date_range 정의

        # 2. 한 그룹 (= entry_date) 에 대한 매매결과 도출
        def process_single_group(group, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop):

            group = group.copy()
            
            #1. 데이터 정리
            
            value_sum = group.xs('value', axis=1, level=1).sum(axis=1)
            group['value_sum'] = value_sum
            group['daily_pnl'] = value_sum.diff().fillna(0)
            group['cum_pnl'] = group['daily_pnl'].cumsum()
            
            # 만약 캘린더 스프레드인 경우 가장 작은 dte 기준으로 모든 전략의 dte 일치
            min_dte = group.xs('dte', axis = 1, level = 1).min(axis = 1, skipna = True).cummin() 
            group['min_dte'] = min_dte.clip(lower = 0) # 혹시 음수가 있을지도 (없어야겠지만) 있는경우 0처리

            # 2. 익손절 적용
            premium = group.loc[:, 'value_sum'].iloc[0]

            if is_complex_pnl: # 복잡한 leg전략이면 profit = 명확한 point 단위
                profit_threshold = np.abs(profit)
                loss_threshold = - np.abs(loss)
            else: # 단순한 leg전략이면 profit = 초기 프리미엄의 X배수만큼 추가 수익 발생시 / 손실 발생시로
                profit_threshold = np.abs(premium * profit)
                loss_threshold = - np.abs(premium * loss)

            def get_earliest_stop_date(group, condition, default_value):  # try_except 가 지저분해서 helper function 정의
                date_list = group.index.get_level_values('date')
                date_list = date_list[condition]
                return date_list[0] if not date_list.empty else default_value
            
            # 1. dte 기반 : dte_stop 보다 크거나 같은 dte 중에서 가장 작은 dte (= 가장 최근 날짜)
            # 이렇게 고친 이유로 dte = 2 다음 휴무등으로 인해 dte = 0 이면 dte 조건은 그냥 pass해버림 => dte= 2에 손절로 변경
            # 다만 반대로 이래 하면 test 시점에서 아직 만기 남은 종목들이 test 하는 당일 기준으로 전부 dte 손절처리됨 => 그냥 감안
            dte_stop = group.index.get_level_values('date')[group['min_dte'] >= dte_stop].max()
            custom_stop = get_earliest_stop_date(group, group.index.get_level_values('date').isin(stop_dates), pd.Timestamp('2099-01-01'))        # 2. hard_stop 기반
            profit_stop= get_earliest_stop_date(group, group['cum_pnl'] >= profit_threshold, pd.Timestamp('2099-01-01'))        # 3. profit_stop 기반
            loss_stop = get_earliest_stop_date(group, group['cum_pnl'] <= loss_threshold, pd.Timestamp('2099-01-01'))        # 4. loss_stop 기반

            if is_intraday_stop == True:

                pass # 장중손절 구현에 대한 추가 코드 작성 필요

            earliest_stop = min(dte_stop, custom_stop, profit_stop, loss_stop)

            stop_values = {
                dte_stop : 'dte',
                custom_stop : 'stop',
                profit_stop : 'win',
                loss_stop : 'loss'
            }

            stop_type = stop_values.get(earliest_stop, None)
            group.loc[:, 'stop'] = np.where(group.index.get_level_values('date') == earliest_stop, stop_type, np.nan)
            group['stop'] = group['stop'].replace('nan', np.nan) # 이상하게 object 타입인 경우 str 'nan' 을 지멋대로 반환함
            
            group = group.loc[: earliest_stop]

            return group
        
        # 3. 위의 개별 그룹에 대한 계산을 병렬처리 또는 모든 그룹들에 대한 적용하는 함수
        def process_groups(df, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop):

            grouped = df.groupby(level = 'entry_date')
            group_count = len(grouped)
            data_size = len(df)

            # 데이터가 크지 않으면 직렬처리, 그룹이 너무 많으면 병렬처리

            if group_count < 200 or data_size < 3000:
                result = grouped.apply(lambda group : process_single_group(group, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop))
                result = result.droplevel(level = 0, axis = 0)

            else:
                result = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(process_single_group)(
                        group, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop
                    ) for _, group in grouped
                )
                result = pd.concat(result)

            pl_columns = [f"{level0}_{level1}" if level1 else level0 for level0, level1 in result.columns]
            result.columns = pl_columns

            return result

        def process_groups_polars(df : pd.DataFrame, stop_dates : typing.Collection, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop):
            """
            벡터화된 그룹 처리 함수 (Polars 기반)
            """
            
            # 사전 컬럼명 정리 후 polar dataframe 으로 변환
            leg_names = df.columns.get_level_values(level = 0).unique() # 전략이름
            df = df.reset_index()
            pl_columns = [f"{level0}_{level1}" if level1 else level0 for level0, level1 in df.columns]
            df.columns = pl_columns
            pl_df = pl.from_pandas(df)
            
            #1. 데이터 정리
            value_columns = leg_names.map(lambda x : x + "_" + 'value')
            dte_columns = leg_names.map(lambda x : x + "_" + 'dte')

            pl_df = pl_df.with_columns(
                pl.sum_horizontal(pl.col(value_columns)).over('entry_date').alias('value_sum'),
                pl.sum_horizontal(pl.col(value_columns)).diff(1).fill_null(0).over('entry_date').alias('daily_pnl'),
                pl.sum_horizontal(pl.col(value_columns)).diff(1).fill_null(0).cum_sum().over('entry_date').alias('cum_pnl'),
                pl.min_horizontal(pl.col(dte_columns)).cum_min().clip(0).over('entry_date').alias('min_dte')
            )
                                    
            # 익절/손절 조건
            pl_df = pl_df.with_columns(
                pl.col('value_sum').first().over('entry_date').alias('premium')
                )
            pl_df = pl_df.with_columns(
                profit_threshold = pl.when(is_complex_pnl == True).then(profit).otherwise((pl.col('premium') * profit).abs()),
                loss_threshold = pl.when(is_complex_pnl == True).then(loss).otherwise(-(pl.col('premium') * loss).abs())
                )

            # 각 조건에 따른 stop date 계산

            # stop_date 계산
            stop_conditions = pl_df.group_by("entry_date").agg([
                # 각 조건에 따른 stop_date 계산
                pl.when(pl.col("cum_pnl") >= pl.col("profit_threshold")).then(pl.col("date")).min().alias("profit_stop"),
                pl.when(pl.col("cum_pnl") <= pl.col("loss_threshold")).then(pl.col("date")).min().alias("loss_stop"),
                pl.when(pl.col("min_dte") >= dte_stop).then(pl.col("date")).max().alias("dte_stop"),
                pl.when(pl.col("date").is_in(stop_dates)).then(pl.col("date")).min().alias("custom_stop"),
            ])

            # 가장 빠른 stop_date 계산
            stop_conditions = stop_conditions.with_columns(
                stop=pl.min_horizontal(["profit_stop", "loss_stop", "dte_stop", "custom_stop"])
            )

            # 왜 stop됬는지 별도 컬럼 생성
            stop_conditions = stop_conditions.with_columns(
                pl.when(pl.col("profit_stop") == pl.col("stop")).then(pl.lit('win'))
                .when(pl.col("loss_stop") == pl.col("stop")).then(pl.lit('loss'))
                .when(pl.col("custom_stop") == pl.col("stop")).then(pl.lit('stop'))
                .when(pl.col("dte_stop") == pl.col("stop")).then(pl.lit('dte'))
                .otherwise(None)
                .alias('whystop')
            )

            # 최종적으로 earliest_stop을 기준으로 필터링
            pl_df = pl_df.join(stop_conditions, how = 'left', left_on = 'entry_date', right_on = 'entry_date')\
                .filter(pl.col("date") <= pl.col("stop"))\
                    .with_columns(
                        pl.when(pl.col('date') == pl.col('stop')).then(pl.col('whystop')).otherwise(None)
                    )

            # 다시 pandas 로 변환하되 컬럼은 싱글컬럼으로 유지
            res = pl_df.to_pandas().set_index(['date', 'entry_date'])
            res['whystop'] = res['whystop'].replace({None : np.nan})

            return res
        
# 도출된 결과 가지고 최종 수익률 등
        
        if use_polars:
            df_result = process_groups_polars(df, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop)
        else:
            df_result = process_groups(df, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop)
        
        df_pnl = df_result['daily_pnl'].to_frame().groupby(level = 'date').sum()
        df_pnl = df_pnl.reindex(date_range).fillna(0)
        df_pnl['cum_pnl'] = df_pnl['daily_pnl'].cumsum()
        df_pnl['dd'] = df_pnl['cum_pnl'] - df_pnl['cum_pnl'].cummax()

        end_time = time.time()

        backtesting_time = end_time - start_time

        # optional : 차트표시 -> verbose = True/False 로 할지말지 만들기
        start_time = time.time()

        if show_chart:
            
            df_k200 = self.k200['close'].reindex(date_range)
            df_vkospi = self.vkospi['close'].reindex(date_range).ffill() # 이상하게 vkospi 에 k200에 있는 일부 영업일이 없는데, 많지 않아서 그냥 ffill() 처리

            fig, axes = plt.subplots(2, 2)
           
            df_pnl['cum_pnl'].plot(ax = axes[0, 0])
            df_k200.plot(ax = axes[0, 0], secondary_y = True) # 1. 누적손익과 기초지수

            df_pnl['cum_pnl'].plot(ax = axes[0, 1])
            df_vkospi.plot(ax = axes[0, 1], secondary_y = True, sharex = True) # 2. 누적손익과 변동성

            df_pnl['dd'].plot(ax = axes[1, 0], kind = 'area')
            df_k200.plot(ax = axes[1, 0], secondary_y = True) # 3. 손실과 기초지수

            df_pnl['dd'].plot(ax = axes[1,1], kind = 'area')
            df_vkospi.plot(ax = axes[1,1], secondary_y = True) # 4. 손실과 변동성

            plt.show()

        end_time = time.time()

        plotting_time = end_time - start_time

        res_dict = {
            'df' : df_result,
            'check' : df_result.loc[:,
                                      (df_result.columns.str.endswith("_name"))|
                                      (df_result.columns.str.endswith("_adj_price"))|
                                      (df_result.columns.str.endswith("_iv"))|
                                      (df_result.columns.str.endswith("min_dte"))|
                                      (df_result.columns.str.endswith("value_sum"))|
                                      (df_result.columns.str.endswith("daily_pnl"))|
                                      (df_result.columns.str.endswith("cum_pnl"))|
                                      (df_result.columns.str.endswith("earliest_stop"))
                                      ],
            'res' : df_result[['cum_pnl', 'stop', 'whystop', 'premium']].loc[df_result['whystop'].dropna().index].sort_values(['cum_pnl'], ascending = True),
            'pnl' : df_pnl
        }

        print(f"backtesting time : {backtesting_time} seconds")
        print(f"plotting time : {plotting_time} seconds")

        return res_dict
        

    @classmethod
    def run_equal_inout(cls, *args : dict):
        res = cls(*args).equal_inout()
        return res


# 2) 진입 lagging / 청산은 한번에 -
# lagged butterfly 전략과 같이 특정 leg 기달렸다가 전체 구축하는 경우가 있으므로 별도로 만들 필요 있음

# 3) 진입 같이 / 청산을 leg별로 따로
# 2번보다 가능성은 낮지만 혹시 single leg 의 청산기준이 전체포지션의 손익구조에 달려있는 전략이 있다면 필요
# 그렇지 않으면 다른 전략 취급

# 4) 진입도 lagging 따로따로 / 청산도 따로따로 -> 그냥 아예 서로 다른 전략 취급...

def get_timeseries(db_path, *args):
    
    conn = sqlite3.connect(db_path)
    res = dict()
    
    for name in args:
        df = pd.read_sql_query(f"SELECT * from {name}", conn, index_col = "date")
        df.index = pd.to_datetime(df.index)
        res[name] = df

    conn.close()

    return res

def get_statistics(df_backtest):

    ''' 
    df_res = get_backtest['res'] 를 말함
    '''

    res = df_backtest['res']
    pnl = df_backtest['pnl']

    mdd = pnl['dd'].min()
    alltime_high = pnl['cum_pnl'].max()
    
    # 각 stop 발생 당일만 모은 sub df
    result = res['stop'].dropna()
    win = result.loc[result['stop'] == 'win']
    loss = result.loc[result['stop'] == 'loss']
    dte = result.loc[result['stop'] == 'dte']
    stop = result.loc[result['stop'] == 'stop']

    # 1) 횟수
    count_all = result.shape[0]
    count_win = win.shape[0]
    count_loss = loss.shape[0]
    count_dte = dte.shape[0]
    count_stop = stop.shape[0]

    # 2) 수익률
    total_win = win['cum_pnl'].sum()
    avg_win = win['cum_pnl'].mean()
    total_loss = loss['cum_pnl'].sum()
    avg_loss = loss['cum_pnl'].mean()
    largest_loss = loss['cum_pnl'].min()
 
# 11

def analyze_iv(df_res):
    iv = df_res.loc[df_res.index == df_res['entry_date']][['iv']]
    other = df_res.loc[df_res['stop'].dropna().index][['entry_date', 'cum_pnl', 'stop']]

    res = pd.merge(iv, other, how = 'left', left_index = True, right_on = 'entry_date')

    win = res.loc[res['stop'] == 'win']
    loss = res.loc[res['stop'] == 'loss']
    dte = res.loc[res['stop'] == 'dte']

    return win, loss, dte

def add_multiple_strat(*args : pd.DataFrame):
    i = 0
    for df in args:
        df_copy = df.copy()
        if i == 0:
            i += 1
            agg_df = df.copy()
        else:
            agg_df = agg_df + df_copy

    agg_df['cum_pnl'] = agg_df['daily_pnl'].cumsum()
    agg_df['dd'] = agg_df['cum_pnl'] - agg_df['cum_pnl'].cummax()

    return agg_df


if __name__ == "__main__":

    db_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_timeseries.db")
    option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/db_option.db")


    df_k200 = get_timeseries(db_path, "k200")['k200']
    entry_dates = df_k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)

#%% weekly_callratio

    var = dict(
    stop_dates = [],
    dte_stop = 1,
    profit = 0.5,
    loss = -2,
    is_complex_pnl = True,
    is_intraday_stop = False,
    start_date = '20100101',
    end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
    show_chart = True,
    use_polars = True
    )

    entry_dates2 = df_k200.weekday(3)
    entry_dates3 = df_k200.weekday(0)

    dict_call1 = {'entry_dates' : entry_dates2, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.2, 'term' : 1, 'volume' : 1}
    dict_call2 = {'entry_dates' : entry_dates2, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.2, 'term' : 1, 'volume' : -2, 'offset' : 2.5}
    callratio1 = backtest(dict_call1, dict_call2).equal_inout(**var)
 
    # dict_call4 = {'entry_dates' : entry_dates3, 'table' : 'weekly_mon', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.2, 'term' : 1, 'volume' : 1}
    # dict_call5 = {'entry_dates' : entry_dates3, 'table' : 'weekly_mon', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.1, 'term' : 1, 'volume' : -2}
    # callratio2 = backtest(dict_call4, dict_call5).equal_inout(**var)


    aggret = callratio1['pnl']
    aggret = aggret.join(df_k200['close'], how = 'left')
    fig, ax = plt.subplots()
    aggret.iloc[:, 0:3].plot(ax = ax)
    aggret['close'].plot(ax = ax, secondary_y= True)

# #%% weekly put backspread
#     var = dict(
#     stop_dates = [],
#     dte_stop = 1,
#     profit = 0.25,
#     loss = -0.5,
#     is_complex_pnl = True,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True
#     )

#     entry_date = df_k200.weekday(0)
#     table = 'weekly_mon'
#     iv_range = [0, 16]
    
#     dict_put1 = {'entry_dates' : entry_date, 'table' : table, 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.3, 'term' : 1, 'volume' : 1, 'iv_range' : iv_range}
#     dict_put2 = {'entry_dates' : entry_date, 'table' : table, 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.15, 'term' : 1, 'volume' : -2, 'iv_range' : iv_range}
    
#     putbs = backtest(dict_put1, dict_put2).equal_inout(**var)

#     aggret = putbs['pnl']
#     aggret = aggret.join(df_k200['close'], how = 'left')
#     fig, ax = plt.subplots()
#     aggret.iloc[:, 0:3].plot(ax = ax)
#     aggret['close'].plot(ax = ax, secondary_y= True)

# #%% weekly buy strangle
#     var = dict(
#     stop_dates = [],
#     dte_stop = 0,
#     profit = 2,
#     loss = -0.3,
#     is_complex_pnl = False,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True
#     )

#     entry_date = df_k200.weekday(1)
#     table = 'weekly_mon'
#     iv_range = [0, 999]
    
#     dict_call = {'entry_dates' : entry_date, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 2.5, 'term' : 1, 'volume' : 1, 'iv_range' : iv_range}
#     dict_put = {'entry_dates' : entry_date, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 2.5, 'term' : 1, 'volume' : 1, 'iv_range' : iv_range}
    
#     buystrg = backtest(dict_call, dict_put).equal_inout(**var)
#     print(buystrg['res'].value_counts('stop'))

#     aggret = buystrg['pnl']
#     aggret = aggret.join(df_k200['close'], how = 'left')
#     fig, ax = plt.subplots()
#     aggret.iloc[:, 0:3].plot(ax = ax)
#     aggret['close'].plot(ax = ax, secondary_y= True)

# #%% weekly_strangle
#     var = dict(
#     stop_dates = [],
#     dte_stop = 1,
#     profit = 0.5,
#     loss = -0.5,
#     is_complex_pnl = False,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True
#     )

#     table = 'weekly_thu'
#     entry_dates = df_k200.weekday(4)
#     iv_range = [13, 999]

#     dict_call1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : -2, 'iv_range' : iv_range}
#     dict_call2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 10, 'term' : 1, 'volume' : 2, 'iv_range' : iv_range}
#     dict_put1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : -2, 'iv_range' : iv_range}
#     dict_put2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 10, 'term' : 1, 'volume' : 2, 'iv_range' : iv_range}

#     straddle = backtest(dict_call1, dict_put1).equal_inout(**var)
#     print(straddle['res']['stop'].value_counts())
#     condor = backtest(dict_call1, dict_call2, dict_put1, dict_put2).equal_inout(**var)
#     print(condor['res']['stop'].value_counts())

#     aggret = straddle['pnl']
#     aggret = aggret.join(df_k200['close'], how = 'left')
#     fig, ax = plt.subplots()
#     aggret.iloc[:, 0:3].plot(ax = ax)
#     aggret['close'].plot(ax = ax, secondary_y= True)

# #%%  weekly_butterfly

#     var = dict(
#     stop_dates = [],
#     dte_stop = 0,
#     profit = 0.5,
#     loss = -999,
#     is_complex_pnl = True,
#     is_intraday_stop = False,
#     start_date = '20120101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True,
#     use_polars = True
#     )

#     entry_dates = df_k200.weekday(4)
#     dte = [1, 999]
#     term = 1
#     table = 'weekly_thu'

#     dict_call1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : term, 'volume' : 1, 'dte' : dte}
#     dict_call2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 10, 'term' : term, 'volume' : -2, 'dte' : dte}
#     dict_call3 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 12.5, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_call1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.15, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_call2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.08, 'term' : term, 'volume' : -2, 'dte' : dte}
#     # dict_call3 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.05, 'term' : term, 'volume' : 1, 'dte' : dte}
    
#     dict_put1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : term, 'volume' : 1, 'dte' : dte}
#     dict_put2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 10, 'term' : term, 'volume' : -2, 'dte' : dte}
#     dict_put3 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 12.5, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_put1 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.15, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_put2 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.08, 'term' : term, 'volume' : -2, 'dte' : dte}
#     # dict_put3 = {'entry_dates' : entry_dates, 'table' : table, 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.05, 'term' : term, 'volume' : 1, 'dte' : dte}
    
#     callbutterfly = backtest(dict_call1, dict_call2, dict_call3).equal_inout(**var)
#     putbutterfly = backtest(dict_put1, dict_put2, dict_put3).equal_inout(**var)

#     aggret = add_multiple_strat(callbutterfly['pnl'], putbutterfly['pnl'])
#     aggret = aggret.join(df_k200['close'], how = 'left')
#     fig, ax = plt.subplots()
#     aggret.iloc[:, 0:3].plot(ax = ax)
#     aggret['close'].plot(ax = ax, secondary_y= True)

# #%% monthly_butterfly -> 이거 되는거 같음 profit 1pt로 콜풋 따로 운용

#     var = dict(
#     stop_dates = [],
#     dte_stop = 0,
#     profit = 1,
#     loss = -999,
#     is_complex_pnl = True,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True,
#     use_polars = True
#     )
  
#     up = df_k200.supertrend.trend('l')
#     down = df_k200.supertrend.trend('s')
#     weekday = df_k200.weekday([0])

#     entry_dates = get_entry_exit.get_date_intersect(weekday)
#     dte = [1, 100]
#     term = 2

#     dict_call1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 20, 'term' : term, 'volume' : 1, 'dte' : dte}
#     dict_call2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 27.5, 'term' : term, 'volume' : -2, 'dte' : dte}
#     dict_call3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 35, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_call1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.15, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_call2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.08, 'term' : term, 'volume' : -2, 'dte' : dte}
#     # dict_call3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.05, 'term' : term, 'volume' : 1, 'dte' : dte}
    
#     dict_put1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 20, 'term' : term, 'volume' : 1, 'dte' : dte}
#     dict_put2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 27.5, 'term' : term, 'volume' : -2, 'dte' : dte}
#     dict_put3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 35, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_put1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.15, 'term' : term, 'volume' : 1, 'dte' : dte}
#     # dict_put2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.08, 'term' : term, 'volume' : -2, 'dte' : dte}
#     # dict_put3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.05, 'term' : term, 'volume' : 1, 'dte' : dte}
    
#     callbutterfly = backtest(dict_call1, dict_call2, dict_call3).equal_inout(**var)
#     putbutterfly = backtest(dict_put1, dict_put2, dict_put3).equal_inout(**var)

#     aggret = add_multiple_strat(callbutterfly['pnl'], putbutterfly['pnl'])
#     aggret = aggret.join(df_k200['close'], how = 'left')
#     fig, ax = plt.subplots()
#     aggret.iloc[:, 0:3].plot(ax = ax)
#     aggret['close'].plot(ax = ax, secondary_y= True)
    

# %%
