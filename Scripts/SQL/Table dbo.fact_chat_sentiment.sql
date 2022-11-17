USE [BI_Malta]
GO

DROP TABLE IF EXISTS dbo.fact_chat_sentiment;

CREATE TABLE [dbo].[fact_chat_sentiment](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[ChatID] [varchar](20) NULL,
	[text] [varchar](max) NULL,
	[UserID] [int] NULL,
	[Date] [datetime] NULL,
	[VIPLevel] [smallint] NULL,
	[Rating] [varchar](20) NULL,
	[DurationInMins] [decimal](12, 2) NULL,
	[Brand] [varchar](50) NULL,
	[CountryGroup1] [varchar](50) NULL,
	[ChatType] [varchar](20) NULL,
	[Negative_Score] [decimal](18, 2) NULL,
	[Neutral_Score] [decimal](18, 2) NULL,
	[Positive_Score] [decimal](18, 2) NULL,
	[Sentiment] [varchar](10) NULL,
	[html_text] [varchar](max) NULL,
	[TransactionDate] [datetime] NULL,
	[AgentDisplayName] [varchar](75) NULL,
 CONSTRAINT [PK__fact_cha__3214EC27595FBC87] PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

ALTER TABLE [dbo].[fact_chat_sentiment] ADD  CONSTRAINT [DF__fact_chat__Trans__0D99FE17]  DEFAULT (getdate()) FOR [TransactionDate]
GO

-- SELECT * FROM dbo.fact_chat_sentiment WITH(NOLOCK)
SELECT COUNT(*) FROM dbo.fact_chat_sentiment WITH(NOLOCK)
-- TRUNCATE TABLE dbo.fact_chat_sentiment

CREATE INDEX IX_UserID_ChatID ON dbo.fact_chat_sentiment (UserID, ChatID)

/*
update b
set [ChatID] = f.id
from  Extractnet_DWH.dbo.dwh_fact_LCListOfChats f 
join [dbo].[fact_chat_sentiment] b on b.userid = f.userid and b.date = f.started
*/
