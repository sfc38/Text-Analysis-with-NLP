import spacy
import spacy.cli
import spacyturk
import pandas as pd
import regex as re
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        pattern = r'\[([^,\]]*?,[^,\]]*?)\]\s*([^:]+):\s*([\s\S]*?)(?=\n\[|$)'
        matches = re.findall(pattern, contents)
        
        # Create a DataFrame from the list of tuples
        df = pd.DataFrame(matches, columns=['datetime_str', 'sender', 'text'])
    return df


def extract_data_from_streamlit_input(uploaded_file):
    data = []
    contents = uploaded_file.read().decode('utf-8')
    pattern = r'\[([^,\]]*?,[^,\]]*?)\]\s*([^:]+):\s*([\s\S]*?)(?=\n\[|$)'
    matches = re.findall(pattern, contents)

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(matches, columns=['datetime_str', 'sender', 'text'])
    return df


def add_datetime_column(df):
    # combine the date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['datetime_str'])

    return df


def add_date_columns(df):
    # add columns for the year, month, day, day of week, and hour of each message
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df['day'] = pd.to_datetime(df['datetime']).dt.day
    df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    return df


def clean_punct_space(text, nlp):
    doc = nlp(text) 
    
    words = []
    for token in doc:
        # exclude punctuation, stop words, spaces
        if not token.is_punct and not token.text.isspace():
            # lemmatize (get the base form of word)
            words.append(token.text)
            
    return words


def clean_punct_space_lemma(text, nlp):
    doc = nlp(text) 
    
    words = []
    for token in doc:
        # exclude punctuation, stop words, spaces
        if not token.is_punct and not token.text.isspace():
            # lemmatize (get the base form of word)
            words.append(token.lemma_)
            
    return words


def clean_punct_stop_space(text, nlp):
    doc = nlp(text) 
    
    words = []
    for token in doc:
        # exclude punctuation, stop words, spaces
        if not token.is_punct and not token.is_stop and not token.text.isspace():
            words.append(token.text)
            
    return words


def clean_punct_stop_space_lemma(text, nlp):
    doc = nlp(text) 
    
    words = []
    for token in doc:
        # exclude punctuation, stop words, spaces
        if not token.is_punct and not token.is_stop and not token.text.isspace():
            # lemmatize (get the base form of word)
            words.append(token.lemma_)
            
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

    # plot the message counts vs time using Plotly
    fig = px.bar(count_by_datetime, x='datetime', y='count', 
                 title='Number of messages across time. (Most active date: {})'.format(most_active_date))

    # set the x-axis tick format to once a month and rotate them by 0 degrees
    fig.update_layout(xaxis=dict(
        tickmode='linear',
        tick0=next((d for d in df['datetime'] if d.day == 1), None),
        dtick='M1',
        tickformat='%B %e, %Y',
        tickangle=0),
        yaxis=dict(title='Number of messages'),
        xaxis_title='Time')

    return fig


def plot_total_messages_by_sender(df):
    # group the DataFrame by sender and count the number of rows in each group
    count_by_sender = df.groupby('sender').size().reset_index(name='count')
    
    # plot the total number of messages per sender using Plotly
    fig = px.bar(count_by_sender, x='sender', y='count', title='Total Number of Messages by Sender')
    return fig


def compute_cumulative_count(df):
    # create a copy of the original DataFrame with just the columns we need
    df_cumulative = df[['datetime', 'sender']].copy()

    # group by sender and date, and count the number of messages for each group
    df_cumulative['count'] = df_cumulative.groupby(['sender', 
                                                    pd.Grouper(key='datetime', 
                                                               freq='D')])['datetime'].transform('count')

    # compute the cumulative sum of messages for each sender and date
    df_cumulative['cumulative_count'] = df_cumulative.groupby(['sender', 
                                                               pd.Grouper(key='datetime', 
                                                                          freq='D')])['count'].cumsum()

    # create a pivot table with dates as the rows and senders as the columns,
    # and the cumulative message counts as the values
    cumulative_count_df = df_cumulative.pivot_table(index=pd.Grouper(key='datetime', freq='D'), 
                                                     columns='sender', 
                                                     values='cumulative_count').fillna(0)

    # compute the cumulative message counts for each sender and day
    cumulative_count_df = cumulative_count_df.cumsum()

    # reset the index and create a new 'date' column
    cumulative_count_df = cumulative_count_df.reset_index().rename(columns={'datetime': 'date'})

    # melt the DataFrame to create a long format
    cumulative_count_df = cumulative_count_df.melt(id_vars=['date'], var_name='sender', value_name='cumulative_count')
    
    return cumulative_count_df


def create_cumulative_count_bar_chart(df):
    
    cumulative_count_df = compute_cumulative_count(df)
    cumulative_count_df['Date'] = cumulative_count_df['date'].dt.date

    # Compute the maximum value of cumulative_count across all senders
    max_cumulative_count = cumulative_count_df['cumulative_count'].max()

    # Set the range_y parameter to a tuple with the lower bound as 0 and the upper bound as the maximum value plus a buffer
    range_y = (0, max_cumulative_count + 10)

    # Create the bar chart with range_y set dynamically
    fig = px.bar(cumulative_count_df, x="sender", y="cumulative_count", color="sender",
      animation_frame="Date", range_y=range_y, title="Cumulative Message Count by Sender")
    
    return fig


def plot_hourly_count_plotly(df):
    # Group messages by hour and count the number of messages for each hour
    hourly_count = df.groupby('hour').size().reset_index(name='count')

    # Create a new DataFrame that contains all hours, even if there are no messages sent at that hour
    all_hours = pd.DataFrame({'hour': range(24)})

    # Merge the hourly_count DataFrame with the all_hours DataFrame to fill in missing hours with a count of 0
    hourly_count = pd.merge(all_hours, hourly_count, on='hour', how='left').fillna(0)

    # Create a bar plot of the hourly count of messages
    fig = px.bar(hourly_count, x='hour', y='count')

    # Set the title of the plot
    fig.update_layout(title_text='Hourly Count of Messages')

    # Set the x-axis label to 'Hour' and the y-axis label to 'Count'
    fig.update_xaxes(title_text='Hour', tickmode='linear', dtick=1)
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
    
    data = get_top_words_by_sender(df, 5, 6)
    
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