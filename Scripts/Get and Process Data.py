import pandas as pd
import numpy as np
import json
import os
import gc
import ast
from tqdm import tqdm
from configparser import ConfigParser

import importlib
from helperFunctions import *
import helperFunctions
from utils import *
import utils
importlib.reload(helperFunctions)
importlib.reload(utils)

# path = 'C:/Users/Jonathanm/Desktop/LiveChat Chats EDA/'

# print(os.getcwd())
# os.chdir('./Scripts')

input_path = '../Input/'
output_path = '../Output/'

currentdate_yyyymmdd = get_current_datetime_num()

config = ConfigParser()
config.read('db.ini')

server = config['dwh']['server']
username = config['dwh']['username']
passwd = config['dwh']['passwd']
database = config['dwh']['database']

print('Configurations:')

print(f'Server: {server}')
print(f'Database: {username}')
print(f'Username: {database}')

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
		CASE WHEN tags like '%chatbot%' THEN 'Chatbot Chat' ELSE 'Support Chat' END AS ChatType,
		Chats.[agents.1.display_name] AS AgentDisplayName
FROM	Extractnet_DWH.dbo.dwh_fact_LCListOfChats AS Chats WITH(NOLOCK)
LEFT JOIN	BI_Malta.dbo.vw_Dim_Profile p ON p.UserID = Chats.UserID
WHERE   started > GETDATE() - 10
  --	started > ISNULL((SELECT MAX(Date) FROM BI_Malta.dbo.fact_chat_sentiment), GETDATE() - 10)
	AND ID IS NOT NULL
"""

logging.info('Getting dataset from DW')

df = pd.read_sql_query(sql, cnxn)

# compress data set
df = reduce_mem_usage(df)

print('Records fetched:', df.shape[0])  # Records fetched: 2109

logging.info(f'Records fetched {df.shape[0]}')

# Get the new data only
# df = df[~df.UserID.isin(dataset.UserID)]

# df = pd.read_csv(input_path + 'Chatbot_Chats_Full_Data_data.csv')
df_excludemessages = pd.read_csv(input_path + 'FilterMessages.csv')

df['DELETE'] = df['Messages'].apply(lambda x: x.endswith('}]'))
# Remove the rows with incomplete Messages
df.query('DELETE == True', inplace=True)

# df.head(2)
# df.shape  # (40885, 10)
# df.columns

df['Messages'] = df['Messages'].apply(lambda x: ast.literal_eval(x))
# pd.json_normalize(df['Messages'][0])  # Works ok


def unpack_transcript(df):
    p = []
    for index, x in tqdm(df.iterrows(), total=len(df)):
        dft = pd.DataFrame(pd.json_normalize(x['Messages']))
        dft['Key'] = index
        p.append(dft)

    p = pd.concat(p).reset_index(drop=True)
    return p


def format_html_text(df):
    ''' This function formats messages and header into HTML code '''
    df['html_header'] = "<b>" + df['author_name'] + \
        " (" + df['user_type'] + ") on " + \
        df['date'] + "</b><br>"

    df['html_text'] = df['html_header'] + df['text'] + "<br><br>"

    df_html = df.groupby(['Key'])['html_text'].apply(
        lambda x: ''.join(x)).reset_index()

    return df_html


USEEXAMPLE = False

if USEEXAMPLE:
    # Example of an angry customer
    filter_user_index = df.query('UserID == 8829912').index
    # df.query('UserID == 8829912').to_csv('./Output/8829912.csv', index = False)

    df = df.copy()
    df = df.loc[filter_user_index]

    df_expanded = unpack_transcript(df)
else:
    df_expanded = unpack_transcript(df)

'''
df_expanded.head(1).T
for c in df_expanded.columns:
    print(c, df_expanded[c].nunique())

df_expanded['user_type'].unique()
df_expanded['agent_id'].unique()

df.iloc[0]
df_expanded.query('Key == 0')
'''

# Get HTML formatted messages
df_html = format_html_text(df_expanded)

# p.user_type.unique()  # 'agent', 'visitor', 'supervisor'
# Remove Customer Support Chat Bot records or agent messages
df_expanded = df_expanded.query('user_type == "visitor"').reset_index(drop=True)
df_expanded = df_expanded[['Key', 'text']]

# Export most common answers
# df_expanded.groupby('text').size().sort_values(ascending=False).head(1000).to_csv(output_path + 'textcounts.csv', index=True)

df_expanded = df_expanded[~df_expanded.text.isin(df_excludemessages.FilterMessages)]

logging.info(f'Filtering out common messages')

# Recombine the text
df_textconcat = df_expanded.groupby(['Key'])['text'].apply(
    lambda x: ','.join(x)).reset_index()

logging.info(f'Concatenate texts')

# Combine with main df
df_final1 = df_textconcat.merge(
    df[['UserID', 'Date', 'VIPLevel', 'Rating',
        'DurationInMins', 'Brand', 'CountryGroup1', 'ChatType', 'ID', 'AgentDisplayName']],
    left_on='Key', right_index=True)

df_final1 = df_final1.merge(df_html, how="left", on="Key")

logging.info(f'Merged data sets to get final1')

df_final1.drop(['Key'], axis=1, inplace=True)

logging.info(f'Save json_expanded')

save_obj(output_path, df_final1, 'json_expanded')

del df_expanded, df_textconcat

gc.collect()

print('Get and process data completed successfully')
logging.info(f'Get and process data completed successfully')


'''
pd.concat([pd.DataFrame(json_normalize(x))
           for x in df['Messages']], ignore_index=True)

pd.concat([json_normalize(df.Messages2[key].apply(lambda x: x.map(json.loads)), record_prefix='chat_', errors='ignore')
           for key in df.index])

pd.concat([pd.DataFrame(json_normalize(list(row['Messages'])))
           for index, row in df.iterrows()], ignore_index=True)

pd.concat([pd.DataFrame(json_normalize(list(row['Messages'])))
           for index, row in df.iterrows()], ignore_index=True)


p.dropna(subset=['agent_id'], inplace=True)  # or p = p[p['agent_id'].notna()]
p.to_csv('./output1.csv', index=False)

'''
