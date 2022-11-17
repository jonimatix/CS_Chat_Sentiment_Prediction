
'''
        x = df.groupby(['VIPLevel', 'Sentiment'],
                       as_index=False).size().to_frame().reset_index()
        x.columns = ['VIPLevel', 'Sentiment', 'No of Chats']
        # x = x.pivot(index='VIPLevel', columns='Sentiment',
        #            values='No of Chats').fillna(0)
        # Move index to a column
        # x.reset_index(level=0, inplace=True)

        fig = plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(y='No of Chats', x="VIPLevel",
                         hue="Sentiment", data=x, ci=None, )
        plt.legend(loc='upper right')
        ax.set_xlabel("VIP Level")
        ax.set_title('No of Chats by VIP Level and Sentiment')
        st.pyplot(fig)
        '''

fig = plt.figure()
sns.set_theme(style="whitegrid")
sns.set_style("whitegrid", {'axes.grid': False})
ax = sns.kdeplot(
    data=x, x="DurationInMins", hue="Sentiment",
    fill=True, common_norm=False, palette="deep", alpha=.5, linewidth=1,)
ax.set_xlabel("Duration (Mins)")
ax.set_title('Density Plot - Chat Duration by Sentiment')

st.pyplot(fig)
