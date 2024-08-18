import sqlite3

def db_connect(local_file_path):
        
    conn = sqlite3.connect(local_file_path)
    return conn

def perform_sql(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)