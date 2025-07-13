#%% 
import pandas as pd
import numpy as np
import sqlite3 # 필요없어보임
import duckdb
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
                    offset = None,
                    term_offset = None,
                    option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db"),
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

    catalog_name = option_path.__str__().split("\\").pop().split(".")[0]
    conn = duckdb.connect(option_path)

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
    
    ordinary_dte = {
        1 : range(1, 35),
        2 : range(1, 63),
        3 : range(1, 98)
    }
    
    #3.
    if offset == None:
        if type == "strike":
            query = f'''
            WITH term_selected_data AS (
            SELECT * 
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {catalog_name}.main.{table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
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
                FROM {catalog_name}.main.{table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date
                ORDER BY m.date ASC;
            '''

        elif type == "moneyness":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {catalog_name}.main.{table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
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
                FROM {catalog_name}.main.{table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date
                ORDER BY m.date ASC;
            '''
        # select 하는 델타 범위에 대한 상충되는 두 경우가 있음
        # 1) 만기가 너무 안 남았거나 / 주가가 저 밑에 있는 경우 : 행사가간 델타 차이가 큼 (0.3타겟인데 0.14 다음 0.42 같은) -> range가 넓어야 매매 나감
        # 2) (급등락으로 인해) 원하는 델타에 행사가가 없는 경우 : 0.2 타겟인데 급등으로 인해 상장된 가장 외가옵션 델타가 0.37 -> range 를 좁혀놔야 이상한 매매 안 나감
        # 결론은 2)번으로 -> 1)번 경우 델타 기반 매매 취지에 안 맞음. 그냥 가장 근접한 델타만 골라서 사는건 X / moneyness 나 pct 기반으로 커버

        elif type == "delta": 
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    ABS(delta - ({select_value})) AS delta_difference,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {catalog_name}.main.{table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                    AND {type} BETWEEN {select_value - 0.1} AND {select_value + 0.1}
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
                FROM {catalog_name}.main.{table} m
                INNER JOIN closest_delta_data d ON m.code = d.code
                WHERE m.date >= d.date
                ORDER BY m.date ASC;
            '''
        
        elif type == "pct":
            query = f'''
            WITH term_selected_data AS (
            SELECT *
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
                    ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
                    FROM {catalog_name}.main.{table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
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
                FROM {catalog_name}.main.{table} m
                INNER JOIN closest_pct_data p ON m.code = p.code
                WHERE m.date >= p.date
                ORDER BY m.date ASC;
            '''

    else: # offset 있는 경우 실질적으로 moneyness/point offset밖에 안 쓸거 같아 이것만 적용함...

        if type == "strike":
            query = f'''
            WITH term_selected_data AS (
            SELECT * 
                FROM
                (SELECT date, iv, code,
                    DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                    FROM {catalog_name}.main.{table}
                    WHERE date IN ('{formatted_dates}')
                    AND cp = '{cp}'
                    AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                    AND {type} = {select_value} + {offset if cp == "C" else -offset}
                    )
                WHERE term = {term_offset}
            ),
            iv_selected_data AS (
                SELECT t.*
                FROM term_selected_data t
                WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
            )
            SELECT m.*, i.term, i.date as entry_date 
                FROM {catalog_name}.main.{table} m
                INNER JOIN iv_selected_data i ON m.code = i.code
                WHERE m.date >= i.date
                ORDER BY m.date ASC;
            '''

        elif type == "moneyness":

            if table == 'monthly':

                exp_selection = f"""
                    STRFTIME(
                        DATE_ADD(
                            CAST(SUBSTR(a.exp, 1, 4) || '-' || SUBSTR(a.exp, 5, 2) || '-01' AS DATE),
                            INTERVAL '{term_offset - term}' MONTH
                        ), 
                        '%Y%m'
                    ) AS new_exp
                    """
                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, iv, cp, exp, strike, dte,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                        AND {type} = {select_value} -- 콜풋 상관없이 moneyness 는 양수일수록 외가
                        )
                    WHERE term = {term}
                ),
                iv_selected_data AS (
                    SELECT t.*
                    FROM term_selected_data t
                    WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
                ),
                new_strike_data AS (
	                SELECT i.date, i.cp, i.exp, 
                        (i.strike + {offset if cp == "C" else -offset}) AS new_strike
	                from iv_selected_data i
                ),
                term_offset_data AS (
                    SELECT m.date, m.cp, m.exp, m.dte, m.strike, {term_offset} as term
                                    FROM {catalog_name}.main.{table} m
                                    INNER JOIN (
                                        SELECT a.date, a.cp,
                                        {exp_selection},
                                        a.new_strike
                                        FROM new_strike_data a
                                    ) offset_data
                                    ON m.date = offset_data.date
                                    AND m.cp = offset_data.cp
                                    AND m.strike = offset_data.new_strike
                                    and m.exp = offset_data.new_exp
                                    where m.dte >= 1
                                    order by m.date
                )
                SELECT m.*, k.term, k.date as entry_date
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN term_offset_data k
                        ON m.cp = k.cp
                        AND m.exp = k.exp
                        AND m.strike = k.strike 
                        WHERE m.date >= k.date
                        ORDER BY m.date ASC;
                '''

            elif table in ['weekly_mon', 'weekly_thu']:

                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, iv, code,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                        AND {type} = {select_value} + {offset} -- 콜풋 상관없이 moneyness 는 양수일수록 외가
                        )
                    WHERE term = {term_offset} 
                ),
                iv_selected_data AS (
                    SELECT t.*
                    FROM term_selected_data t
                    WHERE t.iv BETWEEN {iv_range[0]} AND {iv_range[1]}
                )
                SELECT m.*, i.term, i.date as entry_date 
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN iv_selected_data i ON m.code = i.code
                    WHERE m.date >= i.date
                    ORDER BY m.date ASC;
                '''

        elif type == "delta":

            if table in ['monthly']:
                
                exp_selection = f"""
                    STRFTIME(
                        DATE_ADD(
                            CAST(SUBSTR(a.exp, 1, 4) || '-' || SUBSTR(a.exp, 5, 2) || '-01' AS DATE),
                            INTERVAL '{term_offset - term}' MONTH
                        ), 
                        '%Y%m'
                    ) AS new_exp
                    """
                
                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, cp, exp, strike, iv, dte,
                        ABS(delta - ({select_value})) AS delta_difference,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                        AND {type} BETWEEN {select_value - 0.1} AND {select_value + 0.1}
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
                new_strike_data AS (
	                SELECT d.date, d.cp, d.exp, (d.strike + {offset if cp == "C" else -offset}) AS new_strike
	                from closest_delta_data d
                ),
                term_offset_data AS (
                    SELECT m.date, m.cp, m.exp, m.dte, m.strike, {term_offset} as term
                                    FROM {catalog_name}.main.{table} m
                                    INNER JOIN (
                                        SELECT a.date, a.cp,
                                        {exp_selection},
                                        a.new_strike
                                        FROM new_strike_data a) offset_data
                                    ON m.date = offset_data.date
                                    AND m.cp = offset_data.cp
                                    AND m.strike = offset_data.new_strike
                                    and m.exp = offset_data.new_exp
                                    where m.dte >= 1
                                    order by m.date
                )
                SELECT m.*, k.term, k.date as entry_date
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN term_offset_data k
                        ON m.cp = k.cp
                        AND m.exp = k.exp
                        AND m.strike = k.strike 
                        WHERE m.date >= k.date
                        ORDER BY m.date ASC;
                '''

            elif table in ['weekly_thu', 'weekly_mon']:

                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, cp, exp, strike, iv,
                        ABS(delta - ({select_value})) AS delta_difference,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
                        AND {type} BETWEEN {select_value - 0.1} AND {select_value + 0.1}
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
                ),
                new_strike_data AS (
                SELECT m.date, m.cp, m.exp, m.dte, m.strike
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN (SELECT d.date, d.cp, d.strike + {offset if cp == "C" else -offset} AS new_strike FROM closest_delta_data d) a
                    ON m.date = a.date
                    AND m.cp = a.cp
                    AND m.strike = a.new_strike
                    where m.dte >= {dte[0]} -- dte가 reference 의 dte와 최소값 이상 조건은 동일해야.. 안그럼 dte =0 -> term ->1 로 왜곡되버림
                ),
                term_offset_data AS (
                SELECT *
                    FROM
                    (SELECT n.*,
                        ROW_NUMBER() OVER (PARTITION BY n.date ORDER BY n.dte ASC) AS term
                        FROM new_strike_data n
                    )
                    WHERE term = {term_offset}
                )
                SELECT m.*, k.term, k.date as entry_date
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN term_offset_data k
                    ON m.cp = k.cp
                    AND m.exp = k.exp
                    AND m.strike = k.strike
                    WHERE m.date >= k.date
                    ORDER BY m.date ASC;
                '''
        
        elif type == "pct":

            if table in ['monthly']:
                
                exp_selection = f"""
                    STRFTIME(
                        DATE_ADD(
                            CAST(SUBSTR(a.exp, 1, 4) || '-' || SUBSTR(a.exp, 5, 2) || '-01' AS DATE),
                            INTERVAL '{term_offset - term}' MONTH
                        ), 
                        '%Y%m'
                    ) AS new_exp
                    """

                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, cp, exp, strike, iv, dte,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
                        ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
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
                SELECT t.*
                    FROM iv_selected_data i
                    WHERE i.equal_strikediff_cols = 1
                ),
                new_strike_data AS (
	                SELECT p.date, p.cp, p.exp, (p.strike + {offset if cp == "C" else -offset}) AS new_strike 
	                from closest_pct_data p
                ),
                term_offset_data AS (
                    SELECT m.date, m.cp, m.exp, m.dte, m.strike, {term_offset} as term
                                    FROM {catalog_name}.main.{table} m
                                    INNER JOIN (
                                        SELECT a.date, a.cp,
                                        {exp_selection},
                                        a.new_strike
                                        FROM new_strike_data a) offset_data
                                    ON m.date = offset_data.date
                                    AND m.cp = offset_data.cp
                                    AND m.strike = offset_data.new_strike
                                    and m.exp = offset_data.new_exp
                                    where m.dte >= 1
                                    order by m.date
                )
                SELECT m.*, k.term, k.date as entry_date
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN term_offset_data k
                        ON m.cp = k.cp
                        AND m.exp = k.exp
                        AND m.strike = k.strike 
                        WHERE m.date >= k.date
                        ORDER BY m.date ASC;
                '''

            elif table in ['weekly_thu', 'weekly_mon']:
                
                query = f'''
                WITH term_selected_data AS (
                SELECT *
                    FROM
                    (SELECT date, cp, exp, strike,
                        DENSE_RANK() OVER (PARTITION BY date ORDER BY dte ASC) AS term,
                        ABS(strike - close_k200 * {1 + select_value}) AS strike_difference
                        FROM {catalog_name}.main.{table}
                        WHERE date IN ('{formatted_dates}')
                        AND cp = '{cp}'
                        AND dte BETWEEN {ordinary_dte[term][0] if dte[0] == 1 else dte[0]} and {ordinary_dte[term][-1] if dte[1] == 999 else dte[1]}
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
                SELECT t.*
                    FROM iv_selected_data i
                    WHERE i.equal_strikediff_cols = 1
                ),
                new_strike_data AS (
                SELECT m.date, m.cp, m.exp, m.dte, m.strike
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN (SELECT d.date, d.cp, d.strike + {offset if cp == "C" else -offset} AS new_strike FROM closest_pct_data d) a
                    ON m.date = a.date
                    AND m.cp = a.cp
                    AND m.strike = a.new_strike
                    WHERE m.dte >= {dte[0]} 
                ),
                term_offset_data AS (
                SELECT *
                    FROM
                    (SELECT n.*,
                        ROW_NUMBER() OVER (PARTITION BY n.date ORDER BY n.dte ASC) AS term
                        FROM new_strike_data n
                    )
                    WHERE term = {term_offset}
                )
                SELECT m.*, k.term, k.date as entry_date
                    FROM {catalog_name}.main.{table} m
                    INNER JOIN term_offset_data k
                    ON m.cp = k.cp
                    AND m.exp = k.exp
                    AND m.strike = k.strike
                    WHERE m.date >= k.date
                    ORDER BY m.date ASC;
                '''

    df = conn.execute(query).df().set_index('date')
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

    def __init__(self, *args : dict, option_path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")):

        ''' dte 와 iv 는 optional'''

        start_time = time.time()
        self.option_path = option_path
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
            term_offset = arg.get('term_offset', default_params['term_offset'].default)
            return f"{arg['table']}_{arg['cp']}_{arg['type']}_{arg['select_value']}_{arg['term']}_{arg['volume']}_{dte}_{iv_range}_{offset}_{term_offset}"
        
        def process_arg(arg, default_params):

            if not isinstance(arg, dict):
                raise TypeError(f"arg는 최소 'entry_dates', 'table', 'cp', 'type', 'select_value', 'term', 'volume' 를 키값으로 가지는 dict 여야 함")
            if not self.required_keys.issubset(arg.keys()):
                raise ValueError(f"Missing 최소 required keys : {self.required_keys}")
            if 'offset' in arg.keys() and 'term_offset' not in arg.keys():
                raise ValueError('행사가 offset 으로 진행시 반드시 term_offset 지정 필요')
            
            leg_name = generate_leg_name(arg, default_params)

            arg['option_path'] = self.option_path

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
            
            df.index = pd.to_datetime(df.index, format = "%Y-%m-%d")
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
            
        return concat_df.sort_index(level = 'date', ascending = True)
    
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
                    use_polars = False,
                    apply_costs = True,
                    slippage_point = 0.01,
                    commission_point = 0.002  # 수수료와 슬리피지 적용 여부
                    ):

        start_time = time.time()

        # 1. 추출된 데이터들 전부 시간형으로 변경        
        df = self.concat_df.copy()

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        stop_dates = pd.to_datetime(stop_dates)

        if not df.empty:
            df = df.loc[(slice(start_date, end_date))]
            df = df.loc[start_date <= df.index.get_level_values('entry_date')]
            df_daterange = df.index.get_level_values(level = 'date')
            date_range = self.k200.loc[slice(df_daterange[0], df_daterange[-1])].index
        else:
            date_range = pd.DatetimeIndex([])
            print("Warning: No trades found. Using an empty date_range.")
            return

        def process_single_group(group, stop_dates, dte_stop, profit, loss, is_complex_pnl, is_intraday_stop):
            group = group.copy()
            
            #1. 데이터 정리
            value_sum = group.xs('value', axis=1, level=1).sum(axis=1)
            group['value_sum'] = value_sum
            group['daily_pnl'] = value_sum.diff().fillna(0)
            
            # 수수료와 슬리피지 적용 (고정 포인트)
            if apply_costs:
                # 진입 시 비용 (각 leg별로 고정 포인트 적용)
                entry_cost = 0
                for leg_name in group.columns.get_level_values(0).unique():
                    volume = self.order_volume[leg_name]
                    entry_cost += abs(volume) * (commission_point + slippage_point)
                group.loc[group.index.get_level_values('date') == group.index.get_level_values('entry_date'), 'daily_pnl'] -= entry_cost
                group.loc[group.index.get_level_values('date') == group.index.get_level_values('entry_date'), 'trading_cost'] = entry_cost
                
                # 청산 시 비용 (각 leg별로 고정 포인트 적용)
                exit_cost = 0
                for leg_name in group.columns.get_level_values(0).unique():
                    volume = self.order_volume[leg_name]
                    exit_cost += abs(volume) * (commission_point + slippage_point)
                group.loc[group.index.get_level_values('date') == group.index.get_level_values('date')[-1], 'daily_pnl'] -= exit_cost
                group.loc[group.index.get_level_values('date') == group.index.get_level_values('date')[-1], 'trading_cost'] = exit_cost
            else:
                group['trading_cost'] = 0
            
            group['cum_pnl'] = group['daily_pnl'].cumsum()
            
            # 만약 캘린더 스프레드인 경우 가장 작은 dte 기준으로 모든 전략의 dte 일치
            min_dte = group.xs('dte', axis = 1, level = 1).min(axis = 1, skipna = True).cummin() 
            group['min_dte'] = min_dte.clip(lower = 0)

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
                                    
            # 수수료와 슬리피지 계산
            if apply_costs:
                # 진입/청산 비용 계산 (각 leg별 고정 포인트)
                fixed_cost = sum(abs(vol) * (commission_point + slippage_point) 
                               for vol in self.order_volume.values())
                
                # 진입 시점 비용 적용
                pl_df = pl_df.with_columns([
                    pl.when(pl.col("date") == pl.col("entry_date"))
                    .then(pl.col("daily_pnl") - fixed_cost)
                    .otherwise(pl.col("daily_pnl"))
                    .alias("daily_pnl"),
                    pl.when(pl.col("date") == pl.col("entry_date"))
                    .then(pl.lit(fixed_cost))
                    .otherwise(pl.lit(0))
                    .alias("entry_cost")
                ])
                
                # 누적 손익 재계산 (exit cost 적용 전)
                pl_df = pl_df.with_columns(
                    pl.col("daily_pnl").cum_sum().over("entry_date").alias("cum_pnl")
                )

                # 익절/손절 조건
                pl_df = pl_df.with_columns(
                    pl.col('value_sum').first().over('entry_date').alias('premium')
                )
                pl_df = pl_df.with_columns(
                    profit_threshold = pl.when(is_complex_pnl == True).then(profit).otherwise((pl.col('premium') * profit).abs()),
                    loss_threshold = pl.when(is_complex_pnl == True).then(loss).otherwise(-(pl.col('premium') * loss).abs())
                )

                # stop_date 계산
                stop_conditions = pl_df.group_by("entry_date").agg([
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

                # stop_date를 기준으로 필터링하고 exit cost 적용
                pl_df = pl_df.join(stop_conditions, how = 'left', left_on = 'entry_date', right_on = 'entry_date')\
                    .filter(pl.col("date") <= pl.col("stop"))\
                    .with_columns([
                        # exit cost 적용
                        pl.when(pl.col('date') == pl.col('stop'))
                        .then(pl.col("daily_pnl") - fixed_cost)
                        .otherwise(pl.col("daily_pnl"))
                        .alias("daily_pnl"),
                        pl.when(pl.col('date') == pl.col('stop'))
                        .then(pl.lit(fixed_cost))
                        .otherwise(pl.lit(0))
                        .alias("exit_cost"),
                        # whystop 컬럼 추가
                        pl.when(pl.col('date') == pl.col('stop')).then(pl.col('whystop')).otherwise(None).alias('whystop')
                    ])
                
                # Combine entry and exit costs into total trading cost
                pl_df = pl_df.with_columns(
                    (pl.col("entry_cost") + pl.col("exit_cost")).alias("trading_cost")
                )
                
                # 누적 손익 재계산 (exit cost 적용 후)
                pl_df = pl_df.with_columns(
                    pl.col("daily_pnl").cum_sum().over("entry_date").alias("cum_pnl")
                )
            else:
                pl_df = pl_df.with_columns(pl.lit(0).alias("trading_cost"))

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

        df_check = df_result.loc[:,
                                (df_result.columns.str.endswith("_name"))|
                                (df_result.columns.str.endswith("_strike"))|
                                (df_result.columns.str.endswith("_adj_price"))|
                                (df_result.columns.str.endswith("_iv"))|
                                (df_result.columns.str.endswith("min_dte"))|
                                (df_result.columns.str.endswith("value_sum"))|
                                (df_result.columns.str.endswith("daily_pnl"))|
                                (df_result.columns.str.endswith("cum_pnl"))|
                                (df_result.columns.str.endswith("trading_cost"))|
                                (df_result.columns.str.contains("whystop"))
                                ]
        # k200 종가 추가
        k200_close = df_result.loc[:, df_result.columns.str.endswith('close_k200')].iloc[:, :1]
        k200_close.columns = ['k200']
        df_check = pd.merge(df_check, k200_close, how = 'left', left_index = True, right_index = True)

        df_entry = df_check.loc[df_check.index.get_level_values('date') == df_check.index.get_level_values('entry_date')]
        df_exit = df_check.loc[df_result['whystop'].notna()]

        df_res = df_result[['cum_pnl', 'stop', 'whystop', 'premium']].loc[df_result['whystop'].dropna().index].sort_values(['cum_pnl'], ascending = True)
        df_res['days_taken'] = df_res['stop'] - df_res.index.get_level_values('entry_date')
        df_res['days_taken'] = df_res['days_taken'].dt.days

        res_dict = {
            'df' : df_result,
            'check' : df_check,
            'entry' : df_entry,
            'exit' : df_exit,
            'res' : df_res,
            'pnl' : df_pnl
        }

        print(f"backtesting time : {backtesting_time} seconds")
        print(f"plotting time : {plotting_time} seconds")

        return res_dict
        
    
# 2) 진입 lagging / 청산은 한번에 -
# lagged butterfly 전략과 같이 특정 leg 기달렸다가 전체 구축하는 경우가 있으므로 별도로 만들 필요 있음

# 3) 진입 같이 / 청산을 leg별로 따로
# 2번보다 가능성은 낮지만 혹시 single leg 의 청산기준이 전체포지션의 손익구조에 달려있는 전략이 있다면 필요
# -> 예) 감마스캘핑
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
    
    return None

def get_pivot(cp : typing.Literal['C', 'P'], 
        item : typing.Literal['adj_price', 'iv', 'delta', 'gamma', 'vega'],
        table : typing.Literal['weekly_mon', 'weekly_thu', 'monthly'],
        term = 1,
        path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")
        ):

    conn = duckdb.connect(path)

    if table in ['weekly_mon', 'weekly_thu']:

        df = conn.execute(f"SELECT * FROM weekly_data where cp = '{cp}' and items = '{item}' and table_name = '{table}' and term = {term}").fetchdf().set_index('date')
        df.index = pd.to_datetime(df.index, format = '%Y-%m-%d')
        df_pivot = pd.pivot_table(df, values = 'value', index = [df.index, 'dte', 'k200'], columns = 'moneyness')

    if table == 'monthly':
        
        df = conn.execute(f"SELECT * FROM monthly_data where cp = '{cp}' and items = '{item}' and table_name = '{table}' and term = {term}").fetchdf().set_index('date')
        df.index = pd.to_datetime(df.index, format = '%Y-%m-%d')
        df_pivot = pd.pivot_table(df, values = 'value', index = [df.index, 'dte', 'k200'], columns = 'moneyness')

    conn.close()

    return df_pivot

def get_slope(cp : typing.Literal['C', 'P'],
            item : typing.Literal['adj_price', 'iv', 'delta', 'gamma', 'vega'],
            table : typing.Literal['weekly_mon', 'weekly_thu', 'monthly'],
            term): 

    ''' 변동성 트레이딩 구조에서 동일 콜/풋 내 slope 는 가장 차지하는 비중이 낮다는 판단
    막말로 slope trade인 ratio 와 backspread 도 vega 레벨 자체가 손익에 차지하는 비중이 훨씬 큼
    유사 비유하자면, 금리 하락장 => 듀레이션 롱을 위한 장기물 매수 손익 >>> (기준금리 인하에 따른) 커브 스팁 손익
    (ratio 와 backspread는 사실상 slope trade 라기 보다는 vomma trade 임...)
    '''
    pass

def get_skew(item : typing.Literal['adj_price', 'iv', 'delta', 'gamma', 'vega'],
            table : typing.Literal['weekly_mon', 'weekly_thu', 'monthly'],
            term = 1,
            is_index = True,
            path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")
            ):

    '''skew : 계산식  PUT - CALL IV 로 계산 (대부분 양수로 나오게)'''

    if table in ['weekly_mon', 'weekly_thu']:
        col = [0, 2.5, 5, 7.5, 10]

    else:
        col = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]


    call = get_pivot('C', item, table, term, path)
    put = get_pivot('P', item, table, term, path)

    sub = put.sub(call)

    if is_index == True:
        result = sub[col].mean(axis = 1)
    else:
        result = sub[col]
    return result

def get_calendar(
                cp : typing.Literal['C', 'P'],
                item : typing.Literal['adj_price', 'iv', 'delta', 'gamma', 'vega'],
                ref_table : typing.Literal['weekly_mon', 'weekly_thu', 'monthly'],
                term_1,
                sub_table : typing.Literal['weekly_mon', 'weekly_thu', 'monthly'],
                term_2,
                is_index = True,
                path = pathlib.Path.joinpath(pathlib.Path.cwd().parents[0], "commonDB/option.db")
                ):

    if ref_table in ['weekly_mon', 'weekly_thu']:
        col = [0, 2.5, 5, 7.5, 10]

    else:
        col = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

    ''' 동일방향 (콜-콜 / 풋-풋), 다른 만기 df간의 item 차이 비교'''
    ref_table = get_pivot(cp, item, ref_table, term_1, path)
    sub_table = get_pivot(cp, item, sub_table, term_2, path)

    ref_index = ref_table[col].dropna(how = 'all').index
    ref_table = ref_table.loc[ref_index]
    sub_index = sub_table[col].dropna(how = 'all').index
    sub_table = sub_table.loc[sub_index]

    ref_table = ref_table.reset_index(["dte", "k200"])
    sub_table = sub_table.reset_index(["dte", "k200"])

    df_sub = ref_table.sub(sub_table, axis = 0)[col].dropna(how = 'all', axis = 0)

    df_sub = pd.merge(df_sub, ref_table['dte'], how = 'inner', left_index = True, right_index = True)
    df_sub = pd.merge(df_sub, ref_table['k200'], how = 'inner', left_index = True, right_index = True)

    df_sub = df_sub.set_index([df_sub.index, 'dte', 'k200'])

    if is_index == True:
        result = df_sub[col].mean(axis = 1)

    else:
        result = df_sub[col]

    return result

#%% 

if __name__ == "__main__":

    import itertools

    def get_table(df, table, col_weekly = [0, 2.5, 5, 7.5, 10], col_monthly = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]):
        if table in ['weekly_mon', 'weekly_thu']:
            col = col_weekly
        else:
            col = col_monthly

        df = df[col]
        df = df.mean(axis = 1)
        df_table = df.groupby('dte').describe()
        df_table = df_table.loc[df_table['count'] > 10]
        return df_table
    
    #1. 콜풋iv
    cp = ['C', 'P']
    table = ['weekly_mon', 'weekly_thu', 'monthly']
    term = [1, 2]

    tuples = itertools.product(cp, table, term)

    df_iv= pd.DataFrame()

    for cp, table, term in tuples:

        try:
            iv = get_pivot(cp, 'iv', table, term)
            iv_table = get_table(iv, table)
            iv_table['cp'] = cp
            iv_table['table'] = table
            iv_table['term'] = term
        except KeyError:
            continue

        df_iv = pd.concat([df_iv, iv_table], axis = 0)

    df_iv.to_csv("iv.csv", encoding = 'cp949')

    #2. skew
    cp = ['C', 'P']
    table = ['weekly_mon', 'weekly_thu', 'monthly']
    term = [1, 2]
    tuples = itertools.product(cp, table, term)

    df_skew = pd.DataFrame()

    for cp, table, term in tuples:

        try:
            skew = get_skew('iv', table, term, is_index = False)
            skew_table = get_table(skew, table)
            skew_table['cp'] = cp
            skew_table['table'] = table
            skew_table['term'] = term
        except KeyError:
            continue

        df_skew = pd.concat([df_skew, skew_table], axis = 0)

    df_skew.to_csv('skew.csv', encoding = 'cp949')

    # 3. 양매도 iv
    table = ['weekly_mon', 'weekly_thu', 'monthly']
    term = [1, 2]
    tuples = itertools.product(table, term)

    df_iv_both = pd.DataFrame()

    for table, term in tuples:
        try:
            iv_call = get_pivot('C', 'iv', table = table, term = term)
            iv_put = get_pivot('P', 'iv', table = table, term = term)
            iv_all = (iv_call + iv_put)
            table_all = get_table(iv_all, table)
            table_all['table'] = table
            table_all['term'] = term
        except KeyError:
            continue

        df_iv_both = pd.concat([df_iv_both, table_all], axis = 0)

    df_iv_both.to_csv("iv_both.csv", encoding = 'cp949')

    #4. 캘린더

    cp = ['C', 'P']

    df_calendar = pd.DataFrame()

    for cp in cp:
        calendar1 = get_calendar(cp, 'iv', 'weekly_mon', 1, 'weekly_thu', 1, is_index = False)
        table_1 = get_table(calendar1, 'weekly_mon')
        table_1['cp'] = cp
        table_1['table'] = 'weekly_mon'
        table_1['term'] = 1

        calendar2 = get_calendar(cp, 'iv', 'weekly_thu', 1, 'weekly_mon', 1, is_index = False)
        table_2 = get_table(calendar2, 'weekly_thu')
        table_2['cp'] = cp
        table_2['table'] = 'weekly_thu'
        table_2['term'] = 1

        calendar3 = get_calendar(cp, 'iv', 'monthly', 1, 'monthly', 2, is_index = False)
        table_3 = get_table(calendar3, 'monthly')
        table_3['cp'] = cp
        table_3['table'] = 'monthly'
        table_3['term'] = 1

        calendar4 = get_calendar(cp, 'iv', 'monthly', 2, 'monthly', 1, is_index = False)
        table_4 = get_table(calendar4, 'monthly')
        table_4['cp'] = cp
        table_4['table'] = 'monthly'
        table_4['term'] = 2

        calendar5 = get_calendar(cp, 'iv', 'weekly_mon', 1, 'monthly', 1, is_index = False)
        table_5 = get_table(calendar5, 'weekly_mon')
        table_5['cp'] = cp
        table_5['table'] = 'weekly_mon'
        table_5['term'] = 1

        calendar6 = get_calendar(cp, 'iv', 'weekly_thu', 1, 'monthly', 1, is_index = False)
        table_6 = get_table(calendar6, 'weekly_thu')
        table_6['cp'] = cp
        table_6['table'] = 'weekly_thu'
        table_6['term'] = 1

        df_calendar = pd.concat([df_calendar, table_1, table_2, table_3, table_4, table_5, table_6], axis = 0)

    df_calendar.to_csv("calendar2.csv", encoding = 'cp949')




