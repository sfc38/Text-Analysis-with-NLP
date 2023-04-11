import streamlit as st
from my_functions import *
import pandas as pd
import io
import re
import time
import plotly.io as pio


# st.balloons()
container_setup_load_model = st.container()
sidebar = st.sidebar
container_preprocess = st.container()
container_plots = st.container()
container_stats = st.container()

# container_create_word_frequency_per_person_figure = st.container()
container_download_report = st.container()


with container_setup_load_model:
    st.title('WhatsApp Chat Analysis')
    st.write('Please use left sidebar to upload data.')
       
    if 'is_language_models_loaded' not in st.session_state:    
        # display a loading message while the model is loading
        with st.spinner("Loading language models..."):
            nlp_en, nlp_tr = load_language_models()
            st.session_state.nlp_en = nlp_en
            st.session_state.nlp_tr = nlp_tr     
        st.session_state.is_language_models_loaded = True
        

with sidebar:
    # data from user
    st.subheader('Load your data')
    with st.form(key='my_form'):
        uploaded_file = st.file_uploader("Upload the WhatsApp chat file (.txt file)", type=['txt'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:

            selected_language = st.selectbox("Text Lang.", ["EN", "TR"], key='selected_language')
        
        with col2:
            selected_duration = st.selectbox("Duration", ["Entire Duration", "Last 12 Months", "Current Year", "Last Year"], 
                                        key='selected_duration') 

        submitted = st.form_submit_button('Submit')

        if submitted:
            df = extract_data_from_streamlit_input_all_lines(uploaded_file)
            st.session_state.df = df

            if selected_language == "EN":
                st.session_state.model = st.session_state.nlp_en
                st.session_state.clean_fn = clean_punct_stop_space_lemma
            elif selected_language == "TR":
                st.session_state.model = st.session_state.nlp_tr
                st.session_state.clean_fn = clean_punct_stop_space
       
    # NOTE: You can remove the else part if you don't want the sample data uploaded as default.
        else:
            st.session_state.model = st.session_state.nlp_en
            st.session_state.clean_fn = clean_punct_stop_space_lemma
            file_name = 'sample_chat.txt'
            df = extract_data_from_file_path_all_lines(file_name)
            st.session_state.df = df 
            
    
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
        with st.spinner("Processing data... Please be patient. It may take up to 30 seconds depending on the size of the data. Once ready, scroll down to see the entire analysis."):
             
            df = st.session_state.df
            df = add_datetime_column(df)
            
            df_last_12_months, df_current_year, df_previous_year = select_date_ranges(df)
            
            if selected_duration == "Last 12 Months":
                df = df_last_12_months
            elif selected_duration == "Current Year":
                df = df_current_year
            elif selected_duration == "Last Year":
                df = df_previous_year
                
                if len(df) == 0:
                    st.error("No data availbale for 'Last Year', please select anothe duration option.")
                    st.stop()
                
                
            df = clean_omitted_text(df)
            df = extract_replace_urls(df)
            df = extract_and_replace_emojis(df)
            df = add_date_columns(df)
            df = clean_and_count_words(df, st.session_state.clean_fn, st.session_state.model)
            st.session_state.df = df
        
            
with container_stats:
    if 'df' in st.session_state:
        st.subheader('Group Stats')
        with st.spinner("Loading stats..."):
            
            # Group Stats
            group_stats = create_group_stats_part_1(st.session_state.df)
            group_stats = create_group_stats_part_2(st.session_state.df, group_stats)
            my_dict = group_stats
            
            # Convert the dictionary to a DataFrame and then to a list of dictionaries
            df = pd.DataFrame.from_dict(my_dict, orient='index', columns=['Value'])
            data = df.reset_index().rename(columns={'index': 'Column'}).to_dict('records')
            
            group_data = pd.DataFrame(data)

            # Display the table in Streamlit
            st.table(group_data)

            
            # User Stats
            st.subheader('User Stats')
            df_stats = get_user_stats(st.session_state.df)
            
            #  ['Person', 'Num Messages', 'Max Messages in Day', 'Most Active Date',
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
            
        
            tab1, tab2, tab3 = st.tabs(["User Stats Part 1", "User Stats Part 2", "User Stats Part 3"])
            
            h = len(df_stats)
            if len(df_stats) > 20:
                h = 20
            height = h * 35 + 35
                
            with tab1:
                df_stats_1_s = df_stats_1.style.highlight_max(subset=['Num Messages','Max Messages in Day', 'Total Words', 'Unique Words'])
                st.dataframe(df_stats_1_s, height = height)
                
            with tab2:
                df_stats_2_s = df_stats_2.style.highlight_max(subset=['Words / Message', 'Unique Words / Message'])
                st.dataframe(df_stats_2_s, height = height)
                
            with tab3:
                df_stats_3_s = df_stats_3.style.highlight_max(subset=['Images', 'Audios', 'Videos', 'URLs', 'Emojis', 'Unique Emoji'])
                st.dataframe(df_stats_3_s, height = height)

        
with container_plots:
    with st.spinner("Loading the plots..."):
        if 'df' in st.session_state:
            
            st.subheader('Number of Messages by Date Plot')
            fig1 = plot_message_count_vs_time_bar(st.session_state.df)
            st.plotly_chart(fig1)
            
            
            st.subheader('Messages by Sender Plot')
            tab1, tab2 = st.tabs(["Bar Plot", "Pie Chart"])
            
            with tab1:
                fig2 = plot_total_messages_by_sender(st.session_state.df)
                st.plotly_chart(fig2)    
            with tab2:
                fig3 = plot_total_messages_by_sender_pie(st.session_state.df)
                st.plotly_chart(fig3)


            st.subheader('Number of Messages by Hour Plot')
            fig4 = plot_hourly_count_plotly(st.session_state.df)
            st.plotly_chart(fig4)


            st.subheader("Top Words by Sender Plot")
            n_people = st.session_state.df['sender'].nunique()
            if n_people > 18:
                n_people = 18
            fig5 = create_word_frequency_figure(st.session_state.df, 5, n_people)
            st.plotly_chart(fig5)
            
        
            st.subheader("Group Wordcloud")            
            plt = create_wordcloud(st.session_state.df)
            st.pyplot(plt)
            
     
            st.subheader("Num Messages by Day of Week Plot")            
            fig6 = plot_message_count_by_day_of_week(st.session_state.df)
            st.plotly_chart(fig6)
            
            
            st.subheader("Num of Media Shared by Sender Plot")   
            fig7 = plot_sender_media_stats(st.session_state.df, 9)
            st.plotly_chart(fig7)
            
            
                
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
      
            
with container_download_report:            
    if 'df' in st.session_state:        
        # Convert the DataFrame to an HTML table
        group_html = group_data.to_html(index=False)
        
        table_html_1 = df_stats_1.to_html(index=False)
        table_html_2 = df_stats_2.to_html(index=False)
        table_html_3 = df_stats_3.to_html(index=False)
        
        # Convert plots to HTML
        div1 = pio.to_html(fig1, include_plotlyjs=False)
        div2 = pio.to_html(fig2, include_plotlyjs=False)
        div3 = pio.to_html(fig3, include_plotlyjs=False)
        div4 = pio.to_html(fig4, include_plotlyjs=False)
        div5 = pio.to_html(fig5, include_plotlyjs=False)
        div6 = pio.to_html(fig6, include_plotlyjs=False)
        div7 = pio.to_html(fig7, include_plotlyjs=False)

        template = """
        <html>
        <head>
        <title>WhatsApp Group Chat Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
        <h1>WhatsApp Group Chat Analysis</h1>
        <p>This report is created by <a href="https://sfc38-text-analysis-with-nlp-app-gim2he.streamlit.app/" target="_blank">this Streamlit app</a>.</p>
        {divs}
        </body>
        </html>
        """
        
        # Combine div elements into HTML file
        html = template.format(divs=group_html+table_html_1+table_html_2+table_html_3+div1+div2+div3+div4+div5+div6+div7)

        file_name = "WhatsApp_Chat_Analysis_Report.html"
        st.download_button(label="Download Report", data=html, file_name=file_name, mime="text/html")