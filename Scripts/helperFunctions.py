import pandas as pd
import pandas.io.sql as psql
import numpy as np
import warnings
import logging
import datetime as dt
from datetime import timedelta, datetime
from tqdm import tqdm

from sqlalchemy import create_engine
from configparser import ConfigParser
import pyodbc
import psycopg2
import psycopg2.extras
import os
import gc
import time
import io
import sys
# print(sys.path)

logging.basicConfig(filename='ApplicationLog.log',
                    level=logging.DEBUG,
                    filemode='a',  # append
                    force=True,
                    format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()

    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect_insert(df, table):
    # """Connect to the PostgreSQL database server"""
    conn = None

    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the database...')
        conn = psycopg2.connect(**params)

        print('Executing Query...')

        if len(df) > 0:

            sql = "DELETE FROM " + table + \
                " WHERE TransactionDateID = " + str(df.CurrentDate.max())
            psql.execute(sql, conn)

            print("Deleted any existing records with today's date")

            cur = conn.cursor()

            output = io.StringIO()
            df.to_csv(output, sep='\t', header=False, index=False)

            output.seek(0)
            contents = output.getvalue()

            cur.copy_from(output, table, null="")  # null values become ''

            conn.commit()
            cur.close()

        return "Success"

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()

        return error

    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def connect(sql='SELECT version()'):
    # """Connect to the PostgreSQL database server"""
    conn = None

    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        print('Executing Query...')
        res = psql.read_sql(sql, conn)

        # create a cursor
        # cur = conn.cursor()

        # execute a statement
        # print('Executing Query...')
        # cur.execute(sql)

        # get results
        # result = pd.DataFrame(cur.fetchall())
        # print(result)

        # close the communication with the PostgreSQL
        # cur.close()

        return res

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def connect_insert_sqlserver(df, table, verbose=0):
    """Connect to the SQL Server database server"""
    cnxn = None
    VERBOSE = verbose

    try:
        # read connection parameters
        config = ConfigParser()
        config.read('db.ini')

        server = config['dwh']['server']
        username = config['dwh']['username']
        passwd = config['dwh']['passwd']
        database = config['dwh']['database']

        if VERBOSE:
            print('Configurations:')

            print(f'Server: {server}')
            print(f'Database: {username}')
            print(f'Username: {database}')

        print('Connecting to the database...')
        cnxn = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                              host=server, database=database, user=username, password=passwd)

        if len(df) > 0:

            # print("Deleted any existing records with today's date")
            # sql = "DELETE FROM " + table + \
            #     " WHERE TransactionDateID = " + str(df.CurrentDate.max())

            # Create a cursor from the connection
            cursor = cnxn.cursor()
            print(f'Inserting in {table} {len(df)} records')

            '''
            # Insert Dataframe into SQL Server:
            for index, row in df.iterrows():
                cursor.execute("INSERT INTO {} (UserID, FirstDepositDate, Probability) values(?,?,?)".format(table),
                               row.UserID, row.Prediction_Proba)
            '''

            sql = """MERGE INTO {} AS t
                        USING (
                            VALUES (?,?,?,?,?)
                        ) as s (UserID_newplayers, UserID_vip, distance, Country, RN)
                        ON t.userid = s.UserID_newplayers AND t.RN = s.RN
                        WHEN NOT MATCHED THEN
                            INSERT (UserID, UserID_VIP, Distance, Country, RN)
                            VALUES (UserID_newplayers, UserID_vip, distance, Country, RN);""".format(table)

            for index, row in df.iterrows():
                cursor.execute(sql, row.UserID_newplayers,
                               row.UserID_vip, row.distance, row.Country, row.RN)

            cnxn.commit()
            cursor.close()

        return "Success"

    except (Exception, pyodbc.DatabaseError) as error:
        print(error)
        cnxn.rollback()

        return error

    finally:
        if cnxn is not None:
            cnxn.close()
            print('Database connection closed.')


def connect_insert_audit_sqlserver(df, table, verbose=0):
    """Connect to the SQL Server database server"""
    cnxn = None
    VERBOSE = verbose

    try:
        # read connection parameters
        config = ConfigParser()
        config.read('db.ini')

        server = config['dwh']['server']
        username = config['dwh']['username']
        passwd = config['dwh']['passwd']
        database = config['dwh']['database']

        if VERBOSE:
            print('Configurations:')

            print(f'Server: {server}')
            print(f'Database: {username}')
            print(f'Username: {database}')

        print('Connecting to the database...')
        cnxn = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                              host=server, database=database, user=username, password=passwd)

        if len(df) > 0:

            # print("Deleted any existing records with today's date")
            # sql = "DELETE FROM " + table + \
            #     " WHERE TransactionDateID = " + str(df.CurrentDate.max())

            # Create a cursor from the connection
            cursor = cnxn.cursor()

            print(f'Inserting in {table}')
            # Insert Dataframe into SQL Server:
            for index, row in df.iterrows():
                cursor.execute("INSERT INTO {} (Project, TrainOrPredict, Stat1, Stat1Desc, Stat2, Stat2Desc, Stat3, Stat3Desc, RecordCount, Comments) values(?,?,?,?,?,?,?,?,?,?)".format(table),
                               row.Project, row.TrainOrPredict, row.Stat1, row.Stat1Desc, row.Stat2, row.Stat2Desc, row.Stat3, row.Stat3Desc, row.RecordCount, row.Comments)

            cnxn.commit()
            cursor.close()

        return "Success"

    except (Exception, pyodbc.DatabaseError) as error:
        print(error)
        cnxn.rollback()

        return error

    finally:
        if cnxn is not None:
            cnxn.close()
            print('Database connection closed.')


def connect_insert_chatsentiment_sqlserver(df, table, verbose=0):
    """Connect to the SQL Server database server"""
    cnxn = None
    VERBOSE = verbose

    try:
        # read connection parameters
        config = ConfigParser()
        config.read('db.ini')

        server = config['dwh']['server']
        username = config['dwh']['username']
        passwd = config['dwh']['passwd']
        database = config['dwh']['database']

        if VERBOSE:
            print('Configurations:')

            print(f'Server: {server}')
            print(f'Database: {username}')
            print(f'Username: {database}')

        print('Connecting to the database...')
        cnxn = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                              host=server, database=database, user=username, password=passwd)

        logging.info("Connected to the database")

        if len(df) > 0:

            # print("Deleted any existing records with today's date")
            # sql = "DELETE FROM " + table + \
            #     " WHERE TransactionDateID = " + str(df.CurrentDate.max())

            # Create a cursor from the connection
            cursor = cnxn.cursor()
            cursor.fast_executemany = True

            print(f'Inserting in {table} {len(df)} records')
            logging.info(f'Inserting in {table} {len(df)} records')

            sql = """MERGE INTO {} AS t
                        USING (
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ) as s (
                            ChatID, [text], [UserID], [Date], [VIPLevel], [Rating], [DurationInMins], [Brand],
                            [CountryGroup1], [ChatType], [Negative_Score], [Neutral_Score], [Positive_Score], [Sentiment], html_text
                        )
                        ON t.userid = s.UserID AND t.ChatID = s.ChatID
                        WHEN NOT MATCHED THEN
                            INSERT 
                            ( ChatID, [text], [UserID], [Date], [VIPLevel], [Rating], [DurationInMins], [Brand],
                            [CountryGroup1], [ChatType], [Negative_Score], [Neutral_Score], [Positive_Score], [Sentiment], html_text)
                            VALUES 
                            ( ChatID, [text], [UserID], [Date], [VIPLevel], [Rating], [DurationInMins], [Brand],
                            [CountryGroup1], [ChatType], [Negative_Score], [Neutral_Score], [Positive_Score], [Sentiment], html_text);""".format(table)

            '''
            sql = """INSERT INTO {}
                     ( [text], [UserID], [Date], [VIPLevel], [Rating], [DurationInMins], [Brand],
                       [CountryGroup1], [ChatType], [Negative_Score], [Neutral_Score], [Positive_Score], [Sentiment], html_text)
                    VALUES 
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?);""".format(table)
            '''

            for index, row in df.iterrows():

                cursor.execute(sql,
                               row.ID, row.text, row.UserID, row.Date, row.VIPLevel, row.Rating,
                               row.DurationInMins, row.Brand, row.CountryGroup1, row.ChatType,
                               row.Negative, row.Neutral, row.Positive, row.Sentiment, row.html_text)

            cnxn.commit()
            cursor.close()
            logging.info('Insert completed')

        return "Success"

    except (Exception, pyodbc.DatabaseError) as error:
        print(error)
        logging.error(error)
        logging.error('Failed for record with parameters: ', row.text, row.UserID, row.Date, row.VIPLevel, row.Rating,
                      row.DurationInMins, row.Brand, row.CountryGroup1, row.ChatType,
                      row.Negative, row.Neutral, row.Positive, row.Sentiment)

        cnxn.rollback()

        return error

    finally:
        if cnxn is not None:
            cnxn.close()
            print('Database connection closed.')
