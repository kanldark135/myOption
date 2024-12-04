import pandas as pd
import numpy as np
import sqlite3
import get_entry_exit
import datetime
import time
import typing
import joblib
import matplotlib.pyplot as plt


def get_timeseries(db_path, *args):

    conn = sqlite3.connect(db_path)
    res = dict()
    
    for name in args:
        df = pd.read_sql_query(f"SELECT * from {name}", conn, index_col = "date")
        df.index = pd.to_datetime(df.index)
        res[name] = df

    conn.close()

    return res

# 1. entry row만 뽑아내는 함수 코드

def get_single_entry(entry_date,
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
        )
        SELET *  
            FROM temp_data
            WHERE term = {term}
            LIMIT 1 OFFET 0;
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
            )
            SELECT *
                FROM temp_data
                WHERE term = {term}
                LIMIT 1 OFFSET 0;
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
            )
        SELECT *
            FROM temp_data
            WHERE term = {term}
            LIMIT 1 OFFSET 0;
            '''
    
    elif type == "pct":
        query = f'''
        WITH temp_data AS (
            SELECT *,
                DENSE_RANK() OVER (ORDER BY dte ASC) AS term,
                ABS(strike - close_k200) AS strike_difference
                FROM {table}
                WHERE date = '{entry_date}'
                AND cp = '{cp}'
                AND dte BETWEEN {dte[0]} AND {dte[1]}
                AND strike BETWEEN close_k200 * {1 + select_value} - 1.25 AND close_k200 * {1 + select_value} + 1.25
                    ORDER BY strike_difference ASC
                )
        SELECT *
            FROM temp_data
                WHERE term = {term}
                LIMIT 1 OFFSET 0;
        '''

    df = pd.read_sql(query, conn, index_col = 'date')
    conn.close()

    return df

#2. entry 및 해당 옵션의 만기까지의 전체 로우 뽑아내는 함수

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

    if type == "strike":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (PARTITION BY date ORDER BY date ASC, dte ASC) AS term
            FROM {table}
            WHERE date IN ('{formatted_dates}')
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND {type} = {select_value}
        )
        SELECT m.*, r.term, r.date as entry_date 
            FROM {table} m
            INNER JOIN temp_data r ON m.code = r.code
            WHERE r.term = {term}
            AND m.date >= r.date;
        '''

    elif type == "moneyness":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (PARTITION BY date ORDER BY date ASC, dte ASC) AS term
            FROM {table}
            WHERE date IN ('{formatted_dates}')
            AND cp = '{cp}'
            AND dte between {dte[0]} and {dte[1]}
            AND {type} = {select_value}
        )
        SELECT m.* , r.term, r.date as entry_date
            FROM {table} m
            INNER JOIN temp_data r ON m.code = r.code
            WHERE r.term = {term}
            AND m.date >= r.date;
            '''

    elif type == "delta": # delta between 0.0 and 0.5; 왜냐면 dte = 1~2 + 주가 저 바닥에 200포인트 아래있을때는 2.5pt 차이로 델타가 매우벌어짐
        query = f'''
        WITH temp_data AS (
        SELECT *,
            ABS(delta - ({select_value})) AS delta_difference,
            DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
            FROM {table}
            WHERE date IN ('{formatted_dates}')
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND {type} BETWEEN {select_value - 0.3} AND {select_value + 0.3}
        ),
        sorted_data AS (
        SELECT r.*,
            ROW_NUMBER() OVER (PARTITION BY r.date ORDER BY r.delta_difference ASC) AS equal_deltadiff_cols
            FROM temp_data r
            WHERE r.term = {term}
            AND r.delta_difference = (
                SELECT min(r2.delta_difference)
                    FROM temp_data r2
                    WHERE r2.term = {term}
                    AND r2.date = r.date
                    )
        )
        SELECT m.*, s.term, s.date as entry_date
            FROM {table} m
            INNER JOIN sorted_data s ON m.code = s.code
            WHERE m.date >= s.date
            AND s.equal_deltadiff_cols = 1;
        '''
    
    elif type == "pct":
        query = f'''
        WITH temp_data AS (
        SELECT *,
            DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
            ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
            FROM {table}
            WHERE date IN ('{formatted_dates}')
            AND cp = '{cp}'
            AND dte BETWEEN {dte[0]} AND {dte[1]}
            AND strike BETWEEN close_k200 * {1 + select_value} - 1.25 AND close_k200 * {1 + select_value} + 1.25
        ),
        sorted_data AS (
        SELECT r.*,
            ROW_NUMBER() OVER (PARTITION BY r.date ORDER BY r.strike_difference ASC) AS equal_strikediff_cols
            FROM temp_data r
            WHERE r.term = {term}
            AND r.strike_difference = (
                SELECT min(r2.strike_difference)
                    FROM temp_data r2
                    WHERE r2.term = {term}
                    AND r2.date = r.date
                    )
        )
        SELECT m.*, s.term, s.date as entry_date
            FROM {table} m
            INNER JOIN sorted_data s ON m.code = s.code
            WHERE m.date >= s.date
            AND s.equal_strikediff_cols = 1;
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

        ''' dte 는 optional'''

        required_keys = {'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'volume'}
        dict_raw_df = {}
        dict_order_volume = {}

        start_time = time.time()

        for i, arg in enumerate(args):
            if not isinstance(arg, dict):
                raise TypeError(f" {i+1} 번째 변수가 {'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'dte', 'volume'} 의 딕셔너리가 아님")
            if not required_keys.issubset(arg.keys()):
                raise ValueError(f" {i+1} 번째 변수의 딕셔너리의 키값은 {'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'dte', 'volume'}")
            else:
                # entry date은 빼는 이유 : entry_date 은 구분자가 될 수 없음. 
                # 다른 모든 조건이 동일하다면 entry date 에 따른 차이는 
                # 1) 서로 다른 전략취급하여 백테스팅 따로 하기
                # 2) 만약 같은 전략이라면 get_entry_exit.get_date_union 함수로 여러 entry_date_arrays 들을 하나의 entry_date 취급

                leg_name = f'{arg['table']}_{arg['cp']}_{arg['type']}_{arg['select_value']}_{arg['term']}_{arg['volume']}'
        
                dict_order_volume[leg_name] = arg['volume']

                df = get_option(**arg)
                dict_raw_df[leg_name] = df

        self_order_volume = dict_order_volume
        self.raw_df = dict_raw_df # 1. 각 leg 의 로데이터 모아놓은 dictionary

        def join_legs_on_entry_date(raw_df): # 멀티leg 들 죄다 date/entry_date 두개 기준으로 join 해놓는 모듈러 함수

            concat_df = None
            for key in raw_df.keys():
                df = raw_df[key].copy()
                volume = self_order_volume[key]

                # 1. 각종 작업
                
                df['value'] = df['adj_price'] * volume
                df['delta'] = df['delta'] * volume
                df['gamma'] = df['gamma'] * volume
                df['theta'] = df['theta'] * volume
                df['vega'] = df['vega'] * volume

                df.index = pd.to_datetime(df.index)
                df[['exp_date', 'entry_date']] = df[['exp_date', 'entry_date']].apply(pd.to_datetime, format = '%Y-%m-%d')

                df = df.set_index([df.index, 'entry_date']) # 인덱스를 (date, entry_date) 으로 멀티인덱스화]
                multicol = pd.MultiIndex.from_product([[key], df.columns]) # 컬럼도 leg별 분류를 위해서 멀티컬럼화
                df.columns = multicol

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
        
        self.concat_df = join_legs_on_entry_date(self.raw_df)

        aux_data = get_timeseries("C:/Users/kwan/Desktop/commonDB/db_timeseries.db", 'k200', 'vkospi')
        self.k200 = aux_data['k200']
        self.vkospi = aux_data['vkospi']

        end_time = time.time()

        print(f"importing data : {end_time - start_time} seconds")

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
                    show_chart = False
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
            date_dte_stop = group.index.get_level_values('date')[group['min_dte'] >= dte_stop].max()

            date_hard_stop = get_earliest_stop_date(group, group.index.get_level_values('date').isin(stop_dates), pd.Timestamp('2099-01-01'))        # 2. hard_stop 기반
            date_profit_take= get_earliest_stop_date(group, group['cum_pnl'] >= profit_threshold, pd.Timestamp('2099-01-01'))        # 3. profit_stop 기반
            date_stop_loss = get_earliest_stop_date(group, group['cum_pnl'] <= loss_threshold, pd.Timestamp('2099-01-01'))        # 4. loss_stop 기반

            if is_intraday_stop == True:

                pass # 장중손절 구현에 대한 추가 코드 작성 필요

            earliest_stop = min(date_dte_stop, date_hard_stop, date_profit_take, date_stop_loss)

            stop_values = {
                date_dte_stop : 'dte',
                date_hard_stop : 'stop',
                date_profit_take : 'win',
                date_stop_loss : 'loss'
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

            return result
        
        df_result = process_groups(df, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop)
        
        # V 손익계산에 대한 날짜가 전 영업일이 아니라 해당 옵션이 거래되던 날짜만 포함됨
        # 예) 2019-01-01 ~ 2024-10-30 까지 모든 날짜에 대해 계산되야하는데, 2019-10-11~14 / 2929-02-04~ / 이런식으로 매매하고있던 구간만 계산
        # 어디 date_range 정의해서 같아붙이기

        df_pnl = df_result[['daily_pnl']].groupby(level = 'date').sum()
        df_pnl = df_pnl.reindex(date_range).fillna(0)
        df_pnl['cum_pnl'] = df_pnl['daily_pnl'].cumsum()
        df_pnl['dd'] = df_pnl['cum_pnl'] - df_pnl['cum_pnl'].cummax()

        end_time = time.time()

        backtesting_time = end_time - start_time

        # optional : 차트표시
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
            'dfcheck' : df_result.loc[:, (slice(None), ('name', 'adj_price'))].sort_index(axis =1 , level =0),
            'res' : df_result[['cum_pnl', 'stop']].loc[df_result['stop'].dropna().index].sort_values(['cum_pnl'], ascending = True),
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
# 그냥 single leg 로 따로따로 결과 낸 다

# 3) 진입 같이 / 청산을 leg별로 따로
# 그냥 single leg 로 따로따로 결과 낸 다음 entry_date 기준으로 join

# 4) 진입도 lagging 따로따로 / 청산도 따로따로 -> 그냥 아예 서로 다른 전략 취급...

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


# 해야하는거
    
# V 1) 각 entry 시점의 옵션데이터 불러오기
# 2) 멀티leg 전략의 chaining : 예) "델타 0.2짜리 매수 및 델타 0.2의 행사가를 기준으로 7.5pt / 15pt 동반매수" 전략의 구현
#    - 먼저 제일 첫번째 leg는 정상적으로 도출
#    - 제일 첫번째 leg 에서 도출된 moneyness + @ 값 / strike * pct 값 을 기준으로 다음 leg 의 moneyness 나 strike 도출
#    - 해당 값을 select_value 에 도입하여 나머지 leg 의 

# V 3) 각 entry 시점 기준으로 해당 cp / exp / strike (혹은 그냥 name) 으로 정의되는 옵션데이터 만기까지 (혹은 exit 시점까지)

# V 4) 최종적으로 전기간에 걸쳐 pnl 할때 (=all_pnl) 동일종목에 대해서 같은 날 보유하고있거나 / 같은 날 어떤건 익절 , 어떤건 손절 이렇게 되는데 날짜별로 aggsum 필요
# V 5) 위 result 에서 주요 통계 (언제 어떤 이유로 stop) 등등 미리 적어놔서 통계 function 에서 바로 익절 /손절 / 만기보유 비율 계산할수 있게끔 선작업
# V 6) entry_dates 입력값이 DB상의 dtype 인 str 외에 다른 값 아무렇게나 입력해도 다 받아주게 수정
# (개발계속) 7) def_statistics : 5)번 바탕으로 통계 데이터프레임 만들기 : entry / exit / stop 만 추려서 별도의 인덱스 테이블처럼
# 8) def entry_generator : entry 조건이 복잡하니 보다 손쉽게 입력할 수 있는 wrapper 함수 생각
# 9) 멀티전략 손익 합산해서 보여주는 방향으로 반영 
# 멀티전략의 경우 현재 apply_stop 함수로는 커버 안 됨 -> 멀티전략의 premium 딴에서부터 합산해서 하는 별개 함수 작성 필요

# 9) 현재 point 로 산출되는 손익 % 로 변경 -> 익절 손절조건도 %로 변경해서 scalability 확보
# 10) 익절손절 칠때 장중매매기준으로 하는 경우에 한해 ( = 프리미엄 매도전략들)
#  -> 매도익절의 경우 당일 저가 ~ 당일 종가 사이에 익절레벨 있으면 했다고 침
#  -> 매도손절의 경우 손실이 당일 시가

# 11) 
# 별개분석으로 진입시점 IV별로 주요 매수전략/매도전략 승률 계산해보기 : pivot
# 이것도 무작정 1:1 안되는게...  dte 낮아질수록 자연스레 프리미엄 붙으면서 뻥튀기되는게 있어서... 평소의 IV 수준과 손익비교하기 힘듬

# V 12) ## 델타와의 difference / strikie difference 기준으로 하는 애들 중에 곤란하게도 소숫점 셋째자리까지도 difference 가 일치하는 일부 경우가 포착됨 (2024-09-10 / 델타0.25 / 근월물 풋 진입시 332.5풋은 0.202 / 335풋은 0.298로 차이가 0.48로 동일
# => 쿼리문에 dense_rank() 함수를 difference 작은 순서대로 배열 후 row_number() == 1 처리해서 둘중 하나만 선택되게끔 변경

#V 13) 델타 범위를 너무 좁게 해서 dte 1~2에 주가도 200따리 가있는경우에는 커버가 안됨 (225는 델타 0.1/ 222.5 는 델타 0.43 이런 식)
# => 임의로 델타범위 목표치 +- 0.3으로 넓혀놨음

#V 14) 멀티leg 캘린더 전략의 경우 dte 기준으로 뭘 쓸지?
# => 기본적으로 가장 만기 짧은물건 만기에 다같이 청산한다는 가정 하에 dte.min(axis = 1) 처리해서 사용

# 15) 매도전략의 경우 장중 익손절 구현 : 당일 시고저종 비교해서 굳이 종가에 안 나가도 중간에 손절칠수 있었으면 그가격에 손절(익절) 나간셈 치기

# V 16) 멀티leg 전략에서 특정 leg 의 경우 어떤 기간(주로 15년 이전) 행사가 없는상황
# => 위에 join 함수에서 inner 처리하면 해당 기간 아예 없는 셈 처리

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

    findata_path = "C:/Users/kwan/Desktop/commonDB/db_timeseries.db"
    option_path ="C:/Users/kwan/Desktop/commonDB/db_option.db"


    df_k200 = get_timeseries(findata_path, "k200")['k200']
    entry_dates = df_k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)

# # weekly_callratio

#     var = dict(
#     stop_dates = [],
#     dte_stop = 1,
#     profit = 0.5,
#     loss = -2,
#     is_complex_pnl = True,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True
#     )

#     entry_dates2 = df_k200.weekday(3)
#     entry_dates3 = df_k200.weekday(0)

#     dict_call1 = {'entry_dates' : entry_dates2, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.2, 'term' : 1, 'volume' : 1}
#     dict_call2 = {'entry_dates' : entry_dates2, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.1, 'term' : 1, 'volume' : -2}
    
#     dict_call4 = {'entry_dates' : entry_dates3, 'table' : 'weekly_mon', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.2, 'term' : 1, 'volume' : 1}
#     dict_call5 = {'entry_dates' : entry_dates3, 'table' : 'weekly_mon', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.1, 'term' : 1, 'volume' : -2}
    
#     callratio1 = backtest(dict_call1, dict_call2).equal_inout(**var)
#     callratio2 = backtest(dict_call4, dict_call5).equal_inout(**var)


# # weekly_strangle
#     var = dict(
#     stop_dates = [],
#     dte_stop = 1,
#     profit = 0.25,
#     loss = -0.5,
#     is_complex_pnl = False,
#     is_intraday_stop = False,
#     start_date = '20100101',
#     end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
#     show_chart = True
#     )

#     entry_dates = df_k200.weekday(3)

#     dict_call1 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 5, 'term' : 1, 'volume' : -2}
#     dict_call2 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : 2}
#     dict_put1 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 5, 'term' : 1, 'volume' : -2}
#     dict_put2 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : 2}

#     straddle1 = backtest(dict_call1, dict_put1).equal_inout(**var)
#     print(straddle1['res']['stop'].value_counts())
#     condor1 = backtest(dict_call1, dict_call2, dict_put1, dict_put2).equal_inout(**var)
#     print(condor1['res']['stop'].value_counts())

# weekly_butterfly

    # var = dict(
    # stop_dates = [],
    # dte_stop = 0,
    # profit = 0.75,
    # loss = -0.75,
    # is_complex_pnl = True,
    # is_intraday_stop = False,
    # start_date = '20100101',
    # end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
    # show_chart = True
    # )

    # entry_dates = df_k200.weekday(4)
    # entry_dates2 = df_k200.weekday(0)

    # dict_call1 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : 1}
    # dict_call2 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 10, 'term' : 1, 'volume' : -2}
    # dict_call3 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 12.5, 'term' : 1, 'volume' : 1}
    
    # dict_put1 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 7.5, 'term' : 1, 'volume' : 1}
    # dict_put2 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 10, 'term' : 1, 'volume' : -2}
    # dict_put3 = {'entry_dates' : entry_dates, 'table' : 'weekly_thu', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 12.5, 'term' : 1, 'volume' : 1}
    
    # callbutterfly = backtest(dict_call1, dict_call2, dict_call3).equal_inout(**var)
    # putbutterfly = backtest(dict_put1, dict_put2, dict_put3).equal_inout(**var)


# monthly_butterfly -> 이거 되는거 같음 profit 1pt로 콜풋 따로 운용

    var = dict(
    stop_dates = [],
    dte_stop = 0,
    profit = 1,
    loss = -999,
    is_complex_pnl = True,
    is_intraday_stop = False,
    start_date = '20120101',
    end_date = datetime.datetime.today().strftime("%Y-%m-%d"),
    show_chart = True
    )

    entry_dates = df_k200.weekday(2)

    dict_call1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 15, 'term' : 2, 'volume' : 1}
    dict_call2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 22.5, 'term' : 2, 'volume' : -2}
    dict_call3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'moneyness', 'select_value' : 30, 'term' : 2, 'volume' : 1}
    # dict_call1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.15, 'term' : 2, 'volume' : 1}
    # dict_call2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.08, 'term' : 2, 'volume' : -2}
    # dict_call3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'C', 'type' : 'delta', 'select_value' : 0.05, 'term' : 2, 'volume' : 1}
    
    dict_put1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 15, 'term' : 2, 'volume' : 1}
    dict_put2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 22.5, 'term' : 2, 'volume' : -2}
    dict_put3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'moneyness', 'select_value' : 30, 'term' : 2, 'volume' : 1}
    # dict_put1 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.15, 'term' : 2, 'volume' : 1}
    # dict_put2 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.08, 'term' : 2, 'volume' : -2}
    # dict_put3 = {'entry_dates' : entry_dates, 'table' : 'monthly', 'cp' : 'P', 'type' : 'delta', 'select_value' : -0.05, 'term' : 2, 'volume' : 1}
    
    callbutterfly = backtest(dict_call1, dict_call2, dict_call3).equal_inout(**var)
    putbutterfly = backtest(dict_put1, dict_put2, dict_put3).equal_inout(**var)

    aggret = add_multiple_strat(callbutterfly['pnl'], putbutterfly['pnl'])
    aggret.plot()