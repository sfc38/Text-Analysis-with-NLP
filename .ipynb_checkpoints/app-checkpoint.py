import streamlit as st
from my_functions import *
import pandas as pd
import io
import re
import time

# st.balloons()

container_setup = st.container()
container_load_model = st.container()
sidebar = st.sidebar
container_preprocess = st.container()

container_group_stats = st.container()
container_user_stats = st.container()

container_plot_message_count_vs_time_bar = st.container()
container_bar_plots = st.container()
container_plot_hourly_count_plotly = st.container()
container_word_plots = st.container()
container_create_word_frequency_per_person_figure = st.container()


with container_setup:
    st.title('WhatsApp Chat Analysis')
    st.write('Please use left sidebar to upload data.')
    
    
with container_load_model:       
    if 'is_language_models_loaded' not in st.session_state:    
        # display a loading message while the model is loading
        with st.spinner("Loading language models..."):
            nlp_en, nlp_tr = load_language_models()
            st.session_state.nlp_en = nlp_en
            st.session_state.nlp_tr = nlp_tr     
        # st.success("Language models are loaded.")
        st.session_state.is_language_models_loaded = True
        

with sidebar:
    # data from user
    st.subheader('Load your data')
    with st.form(key='my_form'):
        uploaded_file = st.file_uploader("Upload the WhatsApp chat file (.txt file)", type=['txt'])
        selected_option = st.selectbox("Select the language of the text", ["EN", "TR"]) 
        submitted = st.form_submit_button('Submit')

        if submitted:
            df = extract_data_from_streamlit_input_all_lines(uploaded_file)
            st.session_state.df = df

            if selected_option == "EN":
                st.session_state.model = st.session_state.nlp_en
                st.session_state.clean_fn = clean_punct_stop_space_lemma
            elif selected_option == "TR":
                st.session_state.model = st.session_state.nlp_tr
                st.session_state.clean_fn = clean_punct_stop_space
    
    # sample data
    st.subheader('Load sample data')
    if st.button('Click to load sample data'):
        st.session_state.model = st.session_state.nlp_en
        st.session_state.clean_fn = clean_punct_stop_space_lemma
        file_name = 'sample_chat.txt'
        df = extract_data_from_file_path_all_lines(file_name)
        st.session_state.df = df      
    
    
with container_preprocess:
    if 'df' in st.session_state:
        with st.spinner("Processing data..."):
           
            start_time = time.time()
            
            df = st.session_state.df
            df = clean_omitted_text(df)
            df = extract_replace_urls(df)
            df = extract_and_replace_emojis(df)
            df = add_datetime_column(df)
            df = add_date_columns(df)
            df = clean_and_count_words(df, st.session_state.clean_fn, st.session_state.model)
            st.session_state.df = df
            
            end_time = time.time()
            # st.write(f"Calculation took {end_time - start_time:.2f} seconds")

            
with container_group_stats:
    st.subheader('Group Stats')
    if 'df' in st.session_state:
        with st.spinner("Loading the group stats..."):
            df = st.session_state.df
            group_stats = create_group_stats_part_1(df)
            group_stats = create_group_stats_part_2(df, group_stats)
            my_dict = group_stats
            
            # Convert the dictionary to a DataFrame and then to a list of dictionaries
            df = pd.DataFrame.from_dict(my_dict, orient='index', columns=['Value'])
            data = df.reset_index().rename(columns={'index': 'Column'}).to_dict('records')

            # Display the table in Streamlit
            st.table(pd.DataFrame(data))


with container_user_stats:
    st.subheader('User Stats')
    if 'df' in st.session_state:
        with st.spinner("Loading the user stats..."):
            df = st.session_state.df
            df_stats = get_user_stats(df)
            
           #      ['Person', 'Num Messages', 'Max Messages in Day', 'Most Active Date',
           # 'Total Words', 'Unique Words', 'Words / Message',
           # 'Unique Words / Message', 'Most Active Day of Week', 'Most Active Hour',
           # 'Images', 'Audios', 'Videos', 'URLs', 'Emojis', 'Common Emoji',
           # 'Unique Emoji']

            
            # Split the original dataframe into two halves
            df_stats_1 = df_stats[['Person Name', 'Num Messages', 'Max Messages in Day', 
                                   'Most Active Date', 'Total Words', 'Unique Words']]
            df_stats_2 = df_stats[['Person Name', 'Words / Message', 'Unique Words / Message', 
                                   'Most Active Day of Week', 'Most Active Hour']]
            df_stats_3 = df_stats[['Person Name', 'Images', 'Audios', 'Videos', 
                                   'URLs', 'Emojis', 'Common Emoji', 'Unique Emoji']]

            st.write("User Stats Part 1")
            df_stats_1 = df_stats_1.style.highlight_max(subset=['Num Messages','Max Messages in Day', 'Total Words', 'Unique Words'])
            st.dataframe(df_stats_1, height = len(df_stats)*35 + 35)
            
            st.write("User Stats Part 2")
            df_stats_2 = df_stats_2.style.highlight_max(subset=['Words / Message', 'Unique Words / Message'])
            st.dataframe(df_stats_2, height = len(df_stats)*35 + 35)
            
            st.write("User Stats Part 3")
            df_stats_3 = df_stats_3.style.highlight_max(subset=['Images', 'Audios', 'Videos', 'URLs', 'Emojis', 'Unique Emoji'])
            st.dataframe(df_stats_3, height = len(df_stats)*35 + 35)

        
with container_plot_message_count_vs_time_bar:
    st.subheader('Number of Messages by Date Plot')
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = plot_message_count_vs_time_bar(st.session_state.df)
            st.plotly_chart(fig)
            
            
with container_bar_plots:
    st.subheader('Number of Messages by Sender Plot')
    if 'df' in st.session_state:
        df = st.session_state.df

        with st.spinner("Loading the plot..."):
            fig = plot_total_messages_by_sender(df)
            st.plotly_chart(fig)


        # with st.spinner("Loading the plot..."):
        #     fig = create_cumulative_count_bar_chart(df)
        #     st.plotly_chart(fig)


with container_plot_hourly_count_plotly:
    st.subheader('Number of Messages by Hour Plot')
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = plot_hourly_count_plotly(st.session_state.df)
            st.plotly_chart(fig)

            
with container_word_plots:
    st.subheader("Word Count Plots")            

    if 'df' in st.session_state:
        
        df = st.session_state.df
        
        with st.spinner("Loading the plot..."):
            fig = create_word_frequency_figure(st.session_state.df, 5, 6)
            st.plotly_chart(fig)

        with st.spinner("Loading the plot..."):
            plt = create_wordcloud(st.session_state.df)
            st.pyplot(plt)

                
# with container_create_word_frequency_per_person_figure:
#     st.subheader("Personalized Plots")            
    
#     if 'df' in st.session_state:     
        
#         df = st.session_state.df

#         options = df['sender'].unique()
#         selected_option = st.selectbox('Select a person:', options, key="Top Words")
#         with st.spinner("Loading the plot..."):
#             fig = create_word_frequency_per_person_figure(df, selected_option, 5)
#             st.plotly_chart(fig) 

#         options = df['sender'].unique()
#         selected_option = st.selectbox('Select a person:', options, key="Word Cloud")
#         with st.spinner("Loading the plot..."):
#             plt = create_wordcloud(df, selected_option)
#             st.pyplot(plt)
