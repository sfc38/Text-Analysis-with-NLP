import spacy
import spacy.cli
import spacyturk
import pandas as pd
import numpy as np
import regex as re
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import emoji
import calendar
from datetime import timedelta


@st.cache_data
def load_language_models():
    
    # check if the en_core_web_sm model is already installed
    try:
        nlp_en = spacy.load('en_core_web_sm')
    except OSError:
        # download the en_core_web_sm model from spaCy
        spacy.cli.download('en_core_web_sm')
        nlp_en = spacy.load('en_core_web_sm')

    try:
        nlp_tr = spacy.load('tr_floret_web_lg')
    except OSError:
        # downloads the spaCyTurk model
        spacyturk.download('tr_floret_web_lg')
        nlp_tr = spacy.load('tr_floret_web_lg')
        
    return nlp_en, nlp_tr


def extract_data_from_file_path(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            pattern = r'\[(.*?)\]\s([\w\s]+):\s(.+)'
            # pattern = r'\[(.*?[0-9]{1}.*?[0-9]{2}.*?)\]\s([\w\s]+):\s(.+)'
            match = re.search(pattern, line)
            if match:
                timestamp = match.group(1)
                sender = match.group(2)
                message = match.group(3).replace("\u200e", "")
                data.append([timestamp, sender, message])
            
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['datetime_str', 'sender', 'text'])
    
    return df


def extract_data_from_file_path_all_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            pattern = r'\[(.*?)\]\s([\w\s]+):\s(.+)'
            # pattern = r'\[(.*?[0-9]{1}.*?[0-9]{2}.*?)\]\s([\w\s]+):\s(.+)'
            match = re.search(pattern, line)
            if match:
                if i != 0:
                    data.append([timestamp, sender, message])
                timestamp = match.group(1)
                sender = match.group(2)
                message = match.group(3).replace("\u200e", "")
            else:
                message += ' ' + line
        if message:
            data.append([timestamp, sender, message])
            
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['datetime_str', 'sender', 'text'])
    
    return df


def extract_data_from_streamlit_input(uploaded_file):
    data = []
    for line in uploaded_file:
        pattern = r'\[(.*?)\]\s([\w\s]+):\s(.+)'
        # pattern = r'\[(.*?[0-9]{1}.*?[0-9]{2}.*?)\]\s([\w\s]+):\s(.+)'
        match = re.search(pattern, line.decode('utf-8'))
        if match:
            timestamp = match.group(1)
            sender = match.group(2)
            message = match.group(3).replace("\u200e", "")
            data.append([timestamp, sender, message])
            
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['datetime_str', 'sender', 'text'])
    
    return df


def extract_data_from_streamlit_input_all_lines(uploaded_file):
    data = []
    for i, line in enumerate(uploaded_file):
        pattern = r'\[(.*?)\]\s([\w\s]+):\s(.+)'
        # pattern = r'\[(.*?[0-9]{1}.*?[0-9]{2}.*?)\]\s([\w\s]+):\s(.+)'
        line = line.decode('utf-8')
        match = re.search(pattern, line)
        if match:
            if i != 0:
                data.append([timestamp, sender, message])
            timestamp = match.group(1)
            sender = match.group(2)
            message = match.group(3).replace("\u200e", "")
        else:
            message += ' ' + line
    if message:
        data.append([timestamp, sender, message])
            
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['datetime_str', 'sender', 'text'])
    
    return df


# def clean_omitted_text(df):
#     # Create a new columns
#     df['is_image'] = (df['text'] == 'image omitted')*1
#     df['is_audio'] = (df['text'] == 'audio omitted')*1
#     df['is_video'] = (df['text'] == 'video omitted')*1

#     # Replace the strings 'image omitted', 'audio omitted', or 'video omitted' with an empty string
#     df['text'] = df['text'].str.replace('image omitted', 'image')
#     df['text'] = df['text'].str.replace('audio omitted', 'audio')
#     df['text'] = df['text'].str.replace('video omitted', 'video')
    
#     return df


def clean_omitted_text(df):
    # Create a new columns
    df['is_image'] = df['text'].str.contains(r'image omitted', regex=True)*1
    df['is_audio'] = df['text'].str.contains(r'audio omitted', regex=True)*1
    df['is_video'] = df['text'].str.contains(r'video omitted', regex=True)*1
    df['is_sticker'] = df['text'].str.contains(r'sticker omitted', regex=True)*1

    # Replace the strings 'image omitted', 'audio omitted', or 'video omitted' with an empty string using regex
    df['text'] = df['text'].str.replace(r'image omitted', '', regex=True)
    df['text'] = df['text'].str.replace(r'audio omitted', '', regex=True)
    df['text'] = df['text'].str.replace(r'video omitted', '', regex=True)
    df['text'] = df['text'].str.replace(r'sticker omitted', '', regex=True)
    
    return df



def extract_replace_urls(df):
    # Define a regular expression pattern to match URLs
    url_pattern = r'https?://\S+'

    # Extract all URLs in the 'text' column and store them in a new column 'urls'
    df['urls'] = df['text'].str.findall(url_pattern)

    # Replace all URLs in the 'text' column with the string 'url'
    df['text'] = df['text'].str.replace(url_pattern, '')

    # Count the number of URLs in each row and store them in a new column 'n_urls'
    df['n_urls'] = df['urls'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df


def extract_and_replace_emojis(df):
    # Define a function to extract emojis from a string
    def extract_emojis(text):
        return [char for char in text if emoji.is_emoji(char)]

    # Define a function to replace emojis in a string with ' emoji '
    def replace_emojis(text):
        for char in text:
            if emoji.is_emoji(char):
                text = text.replace(char, '')
        return text

    # Apply the extract_emojis function to the 'text' column and store the result in a new column 'emojis'
    df['emojis'] = df['text'].apply(extract_emojis)

    # Apply the replace_emojis function to the 'text' column to replace emojis with ' emoji '
    df['text'] = df['text'].apply(replace_emojis)
    
    # Count the number of emojis
    df['n_emojis'] = df['emojis'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df


def add_datetime_column(df):
    # combine the date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['datetime_str'])

    return df


def select_date_ranges(df):
    """
    Selects data from the last 12 months, current year, and previous year from the given DataFrame based on the
    specified date column.
    """
    # Find the latest date in the datetime column of the DataFrame
    latest_date = df['datetime'].max().date()

    # Find the start date of the 12-month period
    start_date = (latest_date - timedelta(days=365)).strftime('%Y-%m-%d')

    # Filter the DataFrame to include only the last 12 months of data
    df_last_12_months = df[df['datetime'] >= start_date]

    # Find the year of the latest date
    current_year = latest_date.year

    # Filter the DataFrame to include only the rows from the current year
    df_current_year = df[df['datetime'].dt.year == current_year]

    # Find the year of the previous year
    previous_year = current_year - 1

    # Filter the DataFrame to include only the rows from the previous year
    df_previous_year = df[df['datetime'].dt.year == previous_year]
    
    return df_last_12_months, df_current_year, df_previous_year


def add_date_columns(df):
    # add columns for the year, month, day, day of week, and hour of each message
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df['day'] = pd.to_datetime(df['datetime']).dt.day
    df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['date'] = df['datetime'].dt.date

    return df


def clean_punct_stop_space(text, nlp):
    doc = nlp(text) 
    # Use a list comprehension to filter out punctuation, stop words, and spaces
    words = [token.text.lower() for token in doc
             if not token.is_punct 
             and not token.is_stop 
             and not token.is_digit 
             and not token.like_num
             and not token.text.isspace()]
    return words


def clean_punct_stop_space_lemma(text, nlp):
    doc = nlp(text) 
    # Use a list comprehension to filter out punctuation, stop words, and spaces,
    # and lemmatize the remaining words.
    words = [token.lemma_.lower() for token in doc 
             if not token.is_punct 
             and not token.is_stop 
             and not token.is_digit
             and not token.like_num
             and not token.text.isspace()]
    return words


def clean_and_count_words(df, clean_fn, nlp):
    # Apply the specified clean function to the specified text column of the df DataFrame
    df['clean_words'] = df['text'].apply(lambda x: clean_fn(x, nlp))

    # Length of the words in a message
    df['num_words'] = df['clean_words'].apply(lambda x: len(x))
    
    return df


def plot_message_count_vs_time_bar(df):
    # group the DataFrame by datetime and count the number of rows in each group
    count_by_datetime = df.groupby(pd.Grouper(key='datetime', freq='D')).size().reset_index(name='count')
    
    # get the date with the highest count
    most_active_date_index = count_by_datetime['count'].idxmax()
    most_active_date = count_by_datetime.loc[most_active_date_index, 'datetime'].strftime('%B %e, %Y')

    # determine the duration of the data in days
    duration = (count_by_datetime['datetime'].max() - count_by_datetime['datetime'].min()).days

    # set the x-axis tick format based on the duration of the data
    if duration <= 90:
        tick_format = '%B %e, %Y'
        dtick = 'M1'
    elif duration <= 180:
        tick_format = '%B %Y'
        dtick = 'M2'
    elif duration <= 360:
        tick_format = '%B %Y'
        dtick = 'M3'
    elif duration <= 720:
        tick_format = '%B %Y'
        dtick = 'M6'
    else:
        tick_format = '%Y'
        dtick = 'M12'

    # plot the message counts vs time using Plotly
    fig = px.bar(count_by_datetime, x='datetime', y='count', 
                 title='Number of messages across time. (Most active date: {})'.format(most_active_date),
                 color=count_by_datetime['datetime'].apply(lambda x: 'most_active_day' 
                                                           if x.strftime('%B %e, %Y') == most_active_date else 'other_days'),
                 color_discrete_map={'most_active_day': 'red', 'other_days': 'blue'})

    # set the x-axis tick format and tick marks
    fig.update_layout(xaxis=dict(
        tickmode='linear',
        tick0=next((d for d in df['datetime'] if d.day == 1), None),
        dtick=dtick,
        tickformat=tick_format,
        tickangle=0),
        yaxis=dict(title='Number of messages'),
        xaxis_title='Time')

    return fig


def plot_total_messages_by_sender(df):
    # group the DataFrame by sender and count the number of rows in each group
    count_by_sender = df.groupby('sender').size().reset_index(name='count')
    
    # plot the total number of messages per sender using Plotly
    fig = px.bar(count_by_sender, x='sender', y='count', color='sender',
                 title='Total Number of Messages by Sender', 
                 color_discrete_sequence=px.colors.qualitative.Alphabet)
    return fig


def plot_total_messages_by_sender_pie(df):
    # group the DataFrame by sender and count the number of rows in each group
    count_by_sender = df.groupby('sender').size().reset_index(name='count')
    
    # plot the total number of messages per sender using Plotly
    fig = px.pie(count_by_sender, values='count', names='sender', 
                 title='Percentage of Messages by Sender',
                 color='sender', color_discrete_sequence=px.colors.qualitative.Alphabet)

    # modify the layout of the plot to make it larger
    fig.update_layout(
        width=600,
        height=450
    )

    return fig


def compute_cumulative_count(df):
    # create a copy of the original DataFrame with just the columns we need
    df_cumulative = df[['datetime', 'sender']].copy()

    # group by sender and date, and count the number of messages for each group
    df_cumulative = df_cumulative.groupby(['sender', pd.Grouper(key='datetime', freq='D')]).size().reset_index(name='count')

    # sort by datetime column
    df_cumulative = df_cumulative.sort_values(by='datetime')

    # compute the cumulative sum of messages for each sender and date
    df_cumulative['cumulative_count'] = df_cumulative.groupby(['sender'])['count'].cumsum()

    # pivot the data to create a wide format
    cumulative_count_df = df_cumulative.pivot(index='datetime', columns='sender', values='cumulative_count').fillna(method='ffill')

    # reset the index and create a new 'date' column
    cumulative_count_df = cumulative_count_df.reset_index().rename(columns={'datetime': 'date'})

    # melt the DataFrame to create a long format
    cumulative_count_df = cumulative_count_df.melt(id_vars=['date'], var_name='sender', value_name='cumulative_count')
    
    return cumulative_count_df


def create_cumulative_count_bar_chart(df):
    
    cumulative_count_df = compute_cumulative_count(df)
    cumulative_count_df['Date'] = cumulative_count_df['date'].dt.date
    cumulative_count_df['Month'] = cumulative_count_df['date'].dt.month

    # Compute the maximum value of cumulative_count across all senders
    max_cumulative_count = cumulative_count_df['cumulative_count'].max()

    # Set the range_y parameter to a tuple with the lower bound as 0 and the upper bound as the maximum value plus a buffer
    range_y = (0, max_cumulative_count + 10)

    # Create the bar chart with range_y set dynamically
    fig = px.bar(cumulative_count_df, x="sender", y="cumulative_count", color="sender",
      animation_frame="Date", range_y=range_y, title="Cumulative Message Count by Sender")
    
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1
    
    return fig


def plot_hourly_count_plotly(df):
    # Group messages by hour and count the number of messages for each hour
    hourly_count = df.groupby('hour').size().reset_index(name='count')

    # Create a new DataFrame that contains all hours, even if there are no messages sent at that hour
    all_hours = pd.DataFrame({'hour': range(24)})

    # Merge the hourly_count DataFrame with the all_hours DataFrame to fill in missing hours with a count of 0
    hourly_count = pd.merge(all_hours, hourly_count, on='hour', how='left').fillna(0)

    # Add a column to the DataFrame that indicates whether the hour is AM or PM
    hourly_count['time_of_day'] = ['AM' if hour < 12 else 'PM' for hour in hourly_count['hour']]

    # Create a bar plot of the hourly count of messages, with different colors for AM and PM
    fig = px.bar(hourly_count, x='hour', y='count', color='time_of_day',
                 color_discrete_map={'AM': 'lightblue', 'PM': 'navy'})

    # Set the title of the plot
    fig.update_layout(title_text='Hourly Count of Messages')

    # Set the x-axis label to 'Hour' and the y-axis label to 'Count'
    fig.update_xaxes(title_text='Hour', tickmode='linear', dtick=1,
                     tickvals=hourly_count['hour'], 
                     ticktext=[f'{hour} {time_of_day}' for hour, 
                               time_of_day in zip(hourly_count['hour'], hourly_count['time_of_day'])])
    fig.update_yaxes(title_text='Count')

    return fig


def combine_user_messages(df):
    user_messages_per_user = df.groupby('sender')['clean_words'].apply(lambda x: [word for words in x for word in words])
    
    user_messages_all = []
    for user in user_messages_per_user.index:
        user_messages_all += user_messages_per_user[user]
    
    return user_messages_per_user, user_messages_all


def get_word_counts(words_list):
    return Counter(words_list)


def create_wordcloud(df, user_name='all'):
    user_messages_per_user, user_messages_all = combine_user_messages(df)
    if user_name == 'all':
        data = get_word_counts(user_messages_all)
    else:
        data = get_word_counts(user_messages_per_user[user_name])

    words = data.keys()
    counts = data.values()

    wordcloud = WordCloud(width=800, 
                          height=400, 
                          background_color='white').generate_from_frequencies(dict(zip(words, counts)))

    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # add a  title
    plt.title("Word Cloud for User: {}".format(user_name), fontdict={'fontsize': 12, 'fontweight': 'bold'}, pad=20, loc='left')
    
    return plt


def sort_word_counts(word_counts):
    sorted_counts = sorted(word_counts.items(), key=lambda pair: pair[1], reverse=True)
    return sorted_counts


def get_top_words_by_sender(df, n_words, n_people):
    senders = df['sender'].unique()

    data = {}
    for sender_name in senders:
        user_messages_per_user, user_messages_all = combine_user_messages(df)
        data[sender_name] = dict(sort_word_counts(get_word_counts(user_messages_per_user[sender_name]))[:n_words])

    # Sort the dictionary by sum of values for each person
    sorted_data = sorted(data.items(), key=lambda x: max(x[1].values()), reverse=True)

    # Select the top n_people items
    top_user_words = dict(sorted_data[:n_people])

    return top_user_words


def create_word_frequency_figure(df, n_words, n_people):
    
    data = get_top_words_by_sender(df, n_words, n_people)
    
    def get_subplots_layout(len_data):
        if len_data == 1:
            num_rows = 1
            num_cols = 1
        elif len_data == 2:
            num_rows = 1
            num_cols = 2
        elif len_data == 4:
            num_rows = 2
            num_cols = 2
        else:
            num_rows = (len_data + 2) // 3
            num_cols = 3
        return num_rows, num_cols
    
    num_rows, num_cols = get_subplots_layout(len(data))

    fig_height = 250 * num_rows
    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=list(data.keys()))

    max_count = max([max(data[person].values()) for person in data])

    for i, person in enumerate(data.keys()):
        row = i // num_cols + 1
        col = i % num_cols + 1

        words = list(data[person].keys())
        counts = list(data[person].values())
        
        # Reverse the order of the lists to sort the bars in descending order of count
        words.reverse()
        counts.reverse()

        fig.add_trace(go.Bar(x=counts, y=words, orientation='h', showlegend=False),
                      row=row, col=col)
        fig.update_xaxes(range=[0, max_count], row=row, col=col)
        fig.update_xaxes(title_text='', row=row, col=col)
        fig.update_yaxes(title_text='', row=row, col=col)

    fig.update_layout(title='Top Words', height=fig_height)

    return fig


def create_word_frequency_per_person_figure(df, person, n_words):
    
    df_person = df[df['sender'] == person]
    data = get_top_words_by_sender(df_person, n_words, 1)
    words = list(data[person].keys())
    counts = list(data[person].values())
        
    # Reverse the order of the lists to sort the bars in descending order of count
    words.reverse()
    counts.reverse()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts, y=words, orientation='h', showlegend=False))
    fig.update_layout(title=f'Top Words of {person}')

    return fig


def plot_message_count_by_day_of_week(df):
    # group the DataFrame by day of the week and count the number of messages on each day
    count_by_day = df.groupby('day_of_week').size().reset_index(name='count')

    # create a bar plot of the number of messages by day of the week
    fig = px.bar(count_by_day, x='day_of_week', y='count', 
                 title='Number of Messages Sent by Day of the Week')

    # set the x-axis tick labels to the names of the days of the week
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ))

    # set the colors for the weekdays and weekends
    fig.update_traces(marker=dict(color=['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c']))

    return fig


def plot_sender_media_stats(df, num_people):
    # Load the dataframe
    df_sub = df[['sender', 'n_urls', 'is_image', 'is_audio', 'is_video']]

    # Group the dataframe by 'sender' and sum the relevant columns
    df_grouped = df_sub.groupby('sender').sum()

    # Calculate the row sums of the grouped dataframe and sort by them
    df_grouped = df_grouped.loc[df_grouped.sum(axis=1).sort_values(ascending=False).index]

    # Select the first `num_people` unique senders
    senders = df_grouped.index[:num_people]

    # Define the xtick labels
    xtick_names = ['num url', 'num images', 'num audio', 'num video']

    # Create a subplot of bar plots for each sender
    def get_subplots_layout(num_people):
        if num_people == 1:
            num_rows = 1
            num_cols = 1
        elif num_people == 2:
            num_rows = 1
            num_cols = 2
        elif num_people == 4:
            num_rows = 2
            num_cols = 2
        else:
            num_rows = (num_people + 2) // 3
            num_cols = 3
        return num_rows, num_cols

    num_rows, num_cols = get_subplots_layout(num_people)

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=senders)

    fig_height = 250 * num_rows

    # Initialize the y-axis range to the maximum value across all senders
    y_max = df_grouped.values.max()

    for i, sender in enumerate(senders):
        # Filter the grouped dataframe by the sender
        df_sender = df_grouped.loc[sender]

        # Create a bar trace for the sender
        trace = go.Bar(x=xtick_names, y=df_sender.values, name=sender)

        # Add the bar trace to the subplot
        fig.add_trace(trace, row=(i // 3) + 1, col=(i % 3) + 1)

        # Update the y-axis label for the first subplot in each row
        if i % 3 == 0:
            fig.update_yaxes(title_text='Count', row=(i // 3) + 1, col=1)

        # Set the y-axis range for each subplot to the maximum value across all senders
        fig.update_yaxes(range=[0, y_max], row=(i // 3) + 1, col=(i % 3) + 1)

    # Update the subplot layout
    fig.update_layout(title="Number of Media Shared by Sender", height=fig_height)

    return fig


def create_group_stats_part_1(df):
    # Find the min and max dates in the count_by_datetime DataFrame
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()

    # Calculate the number of days between the min and max dates
    num_days_total = (max_date - min_date).days + 1

    # Number of messages in the group
    num_messages = df.shape[0]

    # Average Numebr of Messages in a Day
    avg_messages_per_day = round(num_messages / num_days_total, 2)

    # Group messages by date and count number of messages in each group
    msg_count = df.groupby(df['datetime'].dt.date).size().reset_index(name='count')

    # Count the number of unique dates with at least one message
    num_days_with_messages = len(msg_count)

    # Percentage of Days with Messages
    percentage = (num_days_with_messages / num_days_total) * 100
    percentage_str = "{:.2f}%".format(round(percentage, 2))

    # Find the row with the highest message count
    most_active_date = msg_count.loc[msg_count['count'].idxmax()]

    # Extract the date and message count
    most_active_date_datetime = most_active_date['datetime']
    most_active_date_count = most_active_date['count']
    
    df_group_stats = {}
    df_group_stats['First Message Date'] = min_date.strftime('%B %e, %Y')
    df_group_stats['Last Message Date'] = max_date.strftime('%B %e, %Y')
    df_group_stats['Num. Days in Group'] = num_days_total
    df_group_stats['Num. Messages in Group'] = num_messages
    df_group_stats['Ave. Num. Messages in Day'] = avg_messages_per_day
    df_group_stats['Num. Days w Messege'] = num_days_with_messages
    df_group_stats['Num. Days w/o Messege'] = num_days_total - num_days_with_messages
    df_group_stats['Percentage of Days w Message'] = percentage_str
    df_group_stats['Max Num. Messages in Day'] = most_active_date_count
    df_group_stats['Most Active Date'] = most_active_date_datetime.strftime('%B %e, %Y')
    
    return df_group_stats


def create_group_stats_part_2(df, df_group_stats):
    
    # Calculate the average number of words per message (no stopwords)
    avg_words_per_message_without_stopwords = np.mean(df['clean_words'].apply(lambda x: len(x)))

    # Calculate the average number of words per message (using the text without removing stopwords)
    avg_words_per_message = np.mean(df['text'].apply(lambda x: len(x.split())))

    # group messages by day of the week and count number of messages in each group
    # extract the day of the week
    msg_count_perday_of_week = df.groupby('day_of_week').size().reset_index(name='count')
    day_of_week = msg_count_perday_of_week.loc[msg_count_perday_of_week['count'].idxmax()]['day_of_week']
    day_of_week_name = calendar.day_name[day_of_week]

    # Group messages by hour and count number of messages in each group
    # find the row with the highest message count
    # extract the hour and message count
    msg_count_by_hour = df.groupby(df['hour']).size().reset_index(name='count')
    most_active_hour = msg_count_by_hour.loc[msg_count_by_hour['count'].idxmax()]
    hour = most_active_hour['hour']

    if hour >= 12:
        am_pm = "PM"
        hour -= 12
    else:
        am_pm = "AM"

    if hour == 0:
        hour = 12

    hour_str = "{:02d}:00 {}".format(hour, am_pm)

    # group messages by sender and count number of messages in each group
    # find the row with the highest message count
    # extract the sender
    msg_count_per_sender = df.groupby('sender').size().reset_index(name='count')
    most_active_person = msg_count_per_sender.loc[msg_count_per_sender['count'].idxmax()]
    most_active_person_sender = most_active_person['sender']

    # group messages by date and sender and count number of messages in each group
    # group messages by date and count number of unique senders in each group
    # find the row with the highest count of unique senders
    # extract the date
    msg_count = df.groupby(['date', 'sender']).size().reset_index(name='count')
    unique_senders_count = msg_count.groupby('date')['sender'].nunique().reset_index(name='count')
    most_active_day = unique_senders_count.loc[unique_senders_count['count'].idxmax()]
    most_active_day_date = most_active_day['date']
    most_active_day_count = most_active_day['count']
    most_active_day_by_sender_str = f"{most_active_day_date.strftime('%B %e, %Y')} with {most_active_day_count} senders"
    
    df_group_stats['Ave. Num. Words in Messsage'] = avg_words_per_message_without_stopwords.round(2)
    df_group_stats['Most Active Day of Week'] = day_of_week_name
    df_group_stats['Most Active Hour in Day'] = hour_str
    df_group_stats['Most Active Person'] = most_active_person_sender
    df_group_stats['Most Active Day by Sender Num'] = most_active_day_by_sender_str

    return df_group_stats


def count_messages_per_sender(df):
    
    # group messages by sender and count number of messages in each group
    msg_count = df.groupby('sender').size().reset_index().rename(columns={0: 'n_messages'})

    return msg_count


def most_active_date_per_sender(df):
    
    # Group messages by sender and date, and count the number of messages in each group
    messages_per_sender_date = df.groupby(['sender', df['datetime'].dt.date]).size()

    # Group the resulting series by sender, and find the maximum count and corresponding date for each sender
    max_messages_per_sender = messages_per_sender_date.groupby('sender').agg(['max', 'idxmax']).reset_index()

    # Rename the idxmax column to date, and extract the date from the tuple
    max_messages_per_sender = max_messages_per_sender.rename(columns={'max':'max_n_messages_day', 
                                                                      'idxmax': 'most_active_date'})
    max_messages_per_sender['most_active_date'] = max_messages_per_sender['most_active_date'].apply(
        lambda x: x[1].strftime('%B %e, %Y'))

    # Return the resulting dataframe
    return max_messages_per_sender


def count_unique_words_per_sender(df):
    
    # Explode the clean_words column to create a new row for each word
    df_exploded = df.explode('clean_words')

    # Count the number of unique words written by each sender
    unique_words_per_sender = df_exploded.groupby('sender')['clean_words']\
    .nunique().reset_index().rename(columns={'clean_words':'n_unique_words'})

    return unique_words_per_sender


def combine_user_messages(df):
    user_messages_per_user = df.groupby('sender')['clean_words'].apply(lambda x: [word for words in x for word in words])
    
    user_messages_all = []
    for user in user_messages_per_user.index:
        user_messages_all += user_messages_per_user[user]
    
    return user_messages_per_user, user_messages_all


def count_words_per_user(df):
    user_messages_per_user, user_messages_all = combine_user_messages(df)
    user_messages_per_user = user_messages_per_user.reset_index()
    n_words = user_messages_per_user['clean_words'].apply(lambda x: len(x))
    n_unique_words = user_messages_per_user['clean_words'].apply(lambda x: len(set(x)))
    result_df = pd.DataFrame({'n_words':n_words, 'n_unique_words':n_unique_words})
    return result_df


def most_active_day_of_week_per_person(df):
    # Group messages by sender and day of the week, and count the number of messages in each group
    most_active_day_per_person = df.groupby(['sender', 'day_of_week']).size().reset_index(name='count')

    # Find the most active day of the week for each sender
    most_active_day_per_person = most_active_day_per_person.loc[most_active_day_per_person
                                                                .groupby('sender')['count'].idxmax()]

    # Replace the day of the week number with the day of the week name
    most_active_day_per_person['day_of_week'] = most_active_day_per_person['day_of_week'].apply(lambda x: 
                                                                                                calendar.day_name[x])

    # Reset the index
    most_active_day_per_person = most_active_day_per_person.reset_index(drop=True)

    return most_active_day_per_person


def most_active_hour_per_person(df):
    # Group messages by sender and hour, and count the number of messages in each group
    most_active_hour_per_person = df.groupby(['sender', 'hour']).size().reset_index(name='count')

    # Find the most active hour for each sender
    most_active_hour_per_person = most_active_hour_per_person.loc[most_active_hour_per_person
                                                                  .groupby('sender')['count'].idxmax()]

    # Convert the hour to a datetime object and format as a string
    most_active_hour_per_person['hour'] = pd.to_datetime(most_active_hour_per_person['hour'], 
                                                         format='%H').dt.strftime('%I:%M %p')

    # Remove leading zero if present
    most_active_hour_per_person['hour'] = most_active_hour_per_person['hour'].apply(lambda x: 
                                                                                    x.lstrip('0') 
                                                                                    if x.startswith('0') else x)

    # Reset the index
    most_active_hour_per_person = most_active_hour_per_person.reset_index(drop=True)

    return most_active_hour_per_person


def count_media_per_sender(df):
    # Count the number of image messages sent by each sender
    image_count = df.groupby('sender')['is_image'].sum()

    # Count the number of audio messages sent by each sender
    audio_count = df.groupby('sender')['is_audio'].sum()

    # Count the number of video messages sent by each sender
    video_count = df.groupby('sender')['is_video'].sum()

    # Create a new DataFrame with the counts of media messages sent by each sender
    result = pd.DataFrame({'image_count': image_count, 
                           'audio_count': audio_count, 
                           'video_count': video_count}).reset_index()

    return result


def count_urls_and_emojis_per_sender(df):
    # Count the number of URLs sent by each sender
    urls_per_sender = df.groupby('sender')['n_urls'].sum()

    # Count the number of emojis sent by each sender
    emojis_per_sender = df.groupby('sender')['n_emojis'].sum()

    result = pd.DataFrame({'urls_per_sender':urls_per_sender, 'emojis_per_sender':emojis_per_sender}).reset_index()
    return result


def get_emoji_stats(df):
    
    # Get a list of emojis per sender
    emojis_per_sender = df.groupby('sender')['emojis'].sum()

    # Get the most common emoji per sender, if there are any emojis
    most_common_emoji_per_sender = emojis_per_sender.apply(lambda x: max(set(x), key=x.count) if len(x) > 0 else None)

    # Get the number of unique emojis sent by each sender
    n_unique_emojis_per_sender = emojis_per_sender.apply(lambda x: len(set(x)))

    result = pd.DataFrame({'most_common_emoji':most_common_emoji_per_sender, 
                           'n_unique_emojis':n_unique_emojis_per_sender}).reset_index()
    
    return result


def get_user_stats(df):

    df_stats = pd.DataFrame()

    df_stats[['Person Name', 
              'Num Messages']] = count_messages_per_sender(df)[['sender', 
                                                                      'n_messages']]

    df_stats[['Max Messages in Day',
              'Most Active Date']] = most_active_date_per_sender(df)[['max_n_messages_day',
                                                                     'most_active_date']]

    # df_stats['Num Unique Words Used'] = count_unique_words_per_sender(df)['n_unique_words']   


    df_stats[['Total Words', 'Unique Words']] = count_words_per_user(df)[['n_words', 'n_unique_words']]

    df_stats['Words / Message'] = round(df_stats['Total Words'] / 
                                              df_stats['Num Messages'] , 2)

    df_stats['Unique Words / Message'] = round(df_stats['Unique Words'] / 
                                                     df_stats['Num Messages'] , 2)

    df_stats['Most Active Day of Week'] = most_active_day_of_week_per_person(df)['day_of_week']

    df_stats['Most Active Hour'] = most_active_hour_per_person(df)['hour']


    df_stats[['Images',
              'Audios',
              'Videos']] = count_media_per_sender(df)[['image_count',
                                                             'audio_count',
                                                             'video_count']]

    df_stats[['URLs', 
              'Emojis']] = count_urls_and_emojis_per_sender(df)[['urls_per_sender', 
                                                                        'emojis_per_sender']]

    df_stats[['Common Emoji', 
              'Unique Emoji']] = get_emoji_stats(df)[['most_common_emoji', 
                                                           'n_unique_emojis']]

    return df_stats

