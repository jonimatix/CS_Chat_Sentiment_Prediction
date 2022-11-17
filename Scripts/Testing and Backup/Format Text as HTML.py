import pandas as pd
import numpy as np
import json
import os
import gc
import ast
from tqdm import tqdm

import importlib
from helperFunctions import *
import helperFunctions
from utils import *
import utils
importlib.reload(helperFunctions)
importlib.reload(utils)

path = 'C:/Users/Jonathanm/Desktop/LiveChat Chats EDA/'
os.chdir(path)

input_path = './Input/'
output_path = './Output/'

currentdate_yyyymmdd = get_current_datetime_num()

config = configparser.ConfigParser()
config.read('db.ini')

server = config['dwh']['server']
username = config['dwh']['username']
passwd = config['dwh']['passwd']
database = config['dwh']['database']

cnxn = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                      host=server, database=database, user=username, password=passwd)

sql = """
SELECT 
		ID,
		Chats.UserID AS UserID,
		p.VIPlevelNo AS VIPLevel,
		Messages, 
		rate AS Rating,
		duration/60 AS DurationInMins,
		p.BrandFriendlyName AS Brand, 
		p.CountryGroup1,
		started AS Date,
		AgentMessagesCount,
		VisitorMessagesCount,
		CASE WHEN tags like '%chatbot%' THEN 'Chatbot Chat' ELSE 'Support Chat' END AS ChatType
FROM	Extractnet_DWH.dbo.dwh_fact_LCListOfChats AS Chats WITH(NOLOCK)
LEFT JOIN	BI_Malta.dbo.vw_Dim_Profile p ON p.UserID = Chats.UserID
WHERE	CAST(started AS DATE) > 
	ISNULL(
		(SELECT MAX(Date) FROM BI_Malta.dbo.fact_chat_sentiment),
		GETDATE() - 30)
	AND ID IS NOT NULL
"""

df = pd.read_sql_query(sql, cnxn)
df = reduce_mem_usage(df)
df = df.query('ChatType == "Support Chat"')
df.head(2)
df.shape  # (869, 12)
df.columns

df['Messages'] = df['Messages'].apply(lambda x: ast.literal_eval(x))


def unpack_transcript(df):
    p = []
    for index, x in tqdm(df.iterrows()):
        dft = pd.DataFrame(pd.json_normalize(x['Messages']))
        dft['Key'] = index
        p.append(dft)

    p = pd.concat(p).reset_index(drop=True)
    return p


df_expanded = unpack_transcript(df)
df_expanded.to_csv(output_path + 'df_expanded.csv', index=False)
# df_expanded

df_example = df_expanded[df_expanded.Key == 1920]

df_example['html_header'] = "<b>" + df_example['author_name'] + \
    " (" + df_example['user_type'] + ") on " + df_example['date'] + "</b><br>"

df_example['html_text'] = df_example['html_header'] + \
    df_example['text'] + "<br><br>"
# df_example['html_text'].values

df_html = df_example.groupby(['Key'])['html_text'].apply(
    lambda x: ''.join(x)).reset_index()

df_html['html_text'][0]
