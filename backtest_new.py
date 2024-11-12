import pandas as pd
import numpy as np
import sqlite3
import get_entry_exit
import datetime
import time
import typing
import joblib

findata_path = "./db_timeseries.db"
option_path ="./option.db"

def get_findata(db_path, *args):

    conn = sqlite3.connect(findata_path)

    res = dict()
    
    for name in args:
        df = pd.read_sql_query(f"SELECT * from {name}", conn, index_col = "date")
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

def get_df(entry_dates,
                    table : typing.Literal['monthly', 'weekly_thu', 'weekly_mon'],
                    cp : typing.Literal['C', 'P'],
                    type : typing.Literal['strike', 'moneyness', 'delta', 'pct'],
                    select_value : int | float,
                    term : typing.Literal[1, 2, 3, 4, 5],
                    dte : list = [1, 999]): # dte = 0으로 두면 만기날 종가에 포지션 들어가는것처럼 되면서 익/손 0짜리 dummy 포지션 생성
    
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

# 멀티leg 고려사항
# 1) 진입일이 같고 exit 도 한번에 하는 경우
# 2) 진입 lagging / 청산은 한번에
# 3) 진입 같이 / 청산을 leg별로 따로
# 4) 진입도 lagging 따로따로 / 청산도 따로따로 -> 그냥 single leg 여러개 합쳐서...

def get_backtest(entry_dates,
                    table : typing.Literal['monthly', 'weekly_thu', 'weekly_mon'],
                    cp : typing.Literal['C', 'P'],
                    type : typing.Literal['strike', 'moneyness', 'delta', 'pct'],
                    select_value : int | float,
                    term : typing.Literal[1, 2, 3, 4, 5],
                    dte : list = [1, 999], # dte = 0으로 두면 만기날 종가에 포지션 들어가는것처럼 되면서 익/손 0짜리 dummy 포지션 생성
                    order = 1,
                    is_complex_pnl = False,
                    is_intraday_stop = False,
                    stop_dates = [],
                    dte_stop = 0,
                    profit = 0,
                    loss = 0,
                    start_date = None,
                    end_dates = datetime.datetime.today().strftime("%Y-%m-%d")
                    ):

    # 1. 데이터의 시작날짜 slicing
    sd = {
        'monthly' : '2008-01-01',
        'weekly_thu' : '2019-09-23',
        'weekly_mon' : '2023-07-31'
        }
    
    if start_date is None:
        start_date = sd[table]

    entry_dates = entry_dates[(entry_dates >= start_date) & (entry_dates <= end_dates)]
    df = get_df(entry_dates, table, cp, type, select_value, term, dte)
    date_range = get_findata(findata_path, 'k200')['k200'].index
    date_range = date_range[(date_range >= start_date) & (date_range <= end_dates)]

    # 2. 추출된 데이터들 전부 시간형으로 변경
    df.index = pd.to_datetime(df.index)
    date_range = pd.to_datetime(date_range)
    df[['exp_date', 'entry_date']] = df[['exp_date', 'entry_date']].apply(pd.to_datetime, format = 'mixed')
    stop_dates = pd.to_datetime(stop_dates)

    # 3. 각 entry ~ 만기까지의 결과 도출

    grouped = df.groupby('entry_date') # 만기로 grouping 하면 만기중에 두번 이상 서로 다른 trade 있으면 겹침

    def single_trade_pnl(group):
        group = group.copy()
        initial_premium = group['adj_price'].iloc[0] * order
        group['premium'] = initial_premium
        group['cum_pnl'] = group['adj_price'] * order - initial_premium
        group['daily_pnl'] = group['cum_pnl'] - group['cum_pnl'].shift(1).fillna(0)

        return group
    
    # joblib 병렬처리 -> 일단 보류
    # num_cores = -1
    # timenow = time.time()
    # result = joblib.Parallel(n_jobs = num_cores)(joblib.delayed(single_trade_pnl)(group) for entry_date, group in grouped)
    # print("time taken : ", time.time() - timenow)

    # 그냥 apply 순서처리
    # timenow = time.time()
    # result = grouped.apply(single_trade_pnl)
    # print("time taken : ", time.time() - timenow)


    # 4. 각 entry ~ 중간 stop까지의 결과로 축소

    def apply_stop(group):
        premium = group['premium'].iloc[0]

        def get_first_date(df, condition, default_value):  # try_except 가 지저분해서 helper function 정의
            indices = df.index[condition]
            return indices[0] if not indices.empty else default_value

        if is_complex_pnl: # 복잡한 leg전략이면 profit = 명확한 point 단위
            profit_threshold = np.abs(profit * order)
            loss_threshold = - np.abs(loss * order)
        else: # 단순한 leg전략이면 profit = 초기 프리미엄의 X배수만큼 추가 수익 발생시 / 손실 발생시로
            profit_threshold = np.abs(premium * profit)
            loss_threshold = - np.abs(premium * loss)

        date_dte_stop = get_first_date(group, group['dte'] == dte_stop, pd.Timestamp('2099-01-01'))        # 1. dte 기반
        date_hard_stop = get_first_date(group, group.index.isin(stop_dates), pd.Timestamp('2099-01-01'))        # 2. hard_stop 기반
        date_profit_take= get_first_date(group, group['cum_pnl'] >= profit_threshold, pd.Timestamp('2099-01-01'))        # 3. profit_stop 기반
        date_stop_loss = get_first_date(group, group['cum_pnl'] <= loss_threshold, pd.Timestamp('2099-01-01'))        # 4. loss_stop 기반

        if is_intraday_stop == True:

            pass # 추가코드작성필요

        earliest_stop = min(date_dte_stop, date_hard_stop, date_profit_take, date_stop_loss)

        stop_values = {
            date_dte_stop : 'dte',
            date_hard_stop : 'stop',
            date_profit_take : 'win',
            date_stop_loss : 'loss'
        }

        stop_type = stop_values.get(earliest_stop, None)
        group['stop'] = np.where(group.index == earliest_stop, stop_type, np.nan)
        group['stop'] = group['stop'].replace('nan', np.nan) # 이상하게 object 타입인 경우 'nan' 을 지멋대로 반환함
        
        group = group.loc[: earliest_stop]

        return group
    
    df_result = grouped.apply(lambda group : group.pipe(single_trade_pnl).pipe(apply_stop))
    df_result = df_result.reset_index(level = ['entry_date'], drop = True)
    # df_result['stop'] = df_result['stop']

    df_pnl = df_result[['daily_pnl']].groupby(df_result.index).sum()
    df_pnl = df_pnl.reindex(date_range, fill_value = 0)
    df_pnl['cum_pnl'] = df_pnl['daily_pnl'].cumsum()
    df_pnl['dd'] = df_pnl['cum_pnl'] - df_pnl['cum_pnl'].cummax()

    res_dict = {
        'res' : df_result,
        'pnl' : df_pnl
    }

    return res_dict


def get_statistics(df_backtest):

    ''' 
    df_res = get_backtest['res'] 를 말함
    '''

    res = df_backtest['res']
    pnl = df_backtest['pnl']

    mdd = pnl['dd'].min()
    alltime_high = pnl['cum_pnl'].max()
    
    # 각 stop 발생 당일만 모은 sub df
    result = res.dropna(subset = 'stop')
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

#V 12)
    ## 델타와의 difference / strikie difference 기준으로 하는 애들 중에
    # 곤란하게도 소숫점 셋째자리까지도 difference 가 일치하는 일부 경우가 포착됨 (2024-09-10 / 델타0.25 / 근월물 풋 진입시 332.5풋은 0.202 / 335풋은 0.298로 차이가 0.48로 동일
    # 이런 경우 random choice 적용하는 쿼리문 추가할 것 -> 해결, 엔트리조건에 term 으로 분류한 뒤 남은애들 재차 delta_difference 로 row_number 나래비세워서 둘중 아무거나 선택

#V 13) 델타 범위를 너무 좁게 해서 dte 1~2에 주가도 200따리 가있는경우에는 커버가 안됨 (225는 델타 0.1/ 222.5 는 델타 0.43 이런 식)
# => 임의로 델타범위 목표치 +- 0.3으로 넓혀놨음

def analyze_iv(df_res):
    iv = df_res.loc[df_res.index == df_res['entry_date']][['iv']]
    other = df_res.loc[df_res['stop'].dropna().index][['entry_date', 'cum_pnl', 'stop']]

    res = pd.merge(iv, other, how = 'left', left_index = True, right_on = 'entry_date')

    win = res.loc[res['stop'] == 'win']
    loss = res.loc[res['stop'] == 'loss']
    dte = res.loc[res['stop'] == 'dte']

    return win, loss, dte


if __name__ == "__main__":

    df_k200 = get_findata(findata_path, "k200")['k200']
    entry_dates = df_k200.stoch.rebound1(pos = 'l', k = 10, d = 5, smooth_d = 5)
    entry_dates2 = df_k200.weekday(0)
    entry_dates3 = df_k200.rsi.rebound(pos = 'l')

    raw_buycall = get_df(entry_dates2, 'monthly', 'C', 'delta', 0.25, 1)
    raw_buycall2 = get_df(entry_dates2, 'monthly', 'C', 'delta', 0.20, 1)

    buycall = get_backtest(entry_dates, 'weekly_thu' , 'C', 'delta', 0.25, 1, order = 1, profit = 2, loss = -0.5)
    sellput = get_backtest(entry_dates2, 'monthly', 'P', 'delta', -0.25, 1, order = -1, profit = 0.25, loss = -2)
    sellcall = get_backtest(entry_dates2, 'monthly', 'C', 'delta', 0.25, 1, order = -1, profit = 0.25, loss = -2)

    b = analyze_iv(sellput['res'])
