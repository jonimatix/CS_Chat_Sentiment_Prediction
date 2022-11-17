d = load_obj(output_path, 'for_html_update')
d = d[['ID', 'html_text']]

doit(df=d, table="BI_Malta.dbo.fact_chat_sentiment")


def doit(df, table, verbose=0):
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

            sql = """UPDATE {}
            SET html_text = ? 
            WHERE ChatID = ?
                AND html_text is null
                """.format(table)

            for index, row in df.iterrows():

                cursor.execute(sql,
                               row.html_text, row.ID)

            cnxn.commit()
            cursor.close()
            logging.info('Insert completed')

        return "Success"

    except (Exception, pyodbc.DatabaseError) as error:
        print(error)
        logging.error(error)

        cnxn.rollback()

        return error

    finally:
        if cnxn is not None:
            cnxn.close()
            print('Database connection closed.')


del d
