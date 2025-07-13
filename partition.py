import duckdb
import os

# 경로 세팅
db_path = "C:/Users/kwan/Desktop/commonDB/option.db"
save_dir = "C:/Users/kwan/Desktop/parquet_files"
os.makedirs(save_dir, exist_ok=True)

# duckdb 연결
con = duckdb.connect(db_path)

# ----------------------------
# 1. monthly : 3년 단위로 parquet 쪼개기
# ----------------------------
year_ranges = [
    ('2007-01-01', '2009-12-31'),
    ('2010-01-01', '2012-12-31'),
    ('2013-01-01', '2015-12-31'),
    ('2016-01-01', '2018-12-31'),
    ('2019-01-01', '2021-12-31'),
    ('2022-01-01', '2025-12-31'),
]

for start, end in year_ranges:
    outfile = f"{save_dir}/monthly_{start[:4]}_{end[:4]}.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM monthly
            WHERE date BETWEEN '{start}' AND '{end}'
        ) TO '{outfile}' 
        (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    print(f"Saved {outfile}")

# ----------------------------
# 2. monthly_data, weekly_mon, weekly_thu, weekly_data 는 통째로 parquet 저장
# ----------------------------
for table in ['monthly_data', 'weekly_mon', 'weekly_thu', 'weekly_data']:
    outfile = f"{save_dir}/{table}.parquet"
    con.execute(f"""
        COPY (SELECT * FROM {table})
        TO '{outfile}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    print(f"Saved {outfile}")

con.close()