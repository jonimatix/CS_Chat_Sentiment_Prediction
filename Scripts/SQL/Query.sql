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
