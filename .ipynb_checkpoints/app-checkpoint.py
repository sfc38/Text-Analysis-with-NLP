import streamlit as st
import my_functions
import pandas as pd
import io
import re

sidebar = st.sidebar
container_setup = st.container()
container_load_model = st.container()
container_upload_file = st.container()
container_select_language = st.container()
container_preprocess = st.container()
container_plot_message_count_vs_time_bar = st.container()
container_plot_total_messages_by_sender = st.container()
container_create_cumulative_count_bar_chart = st.container()
container_plot_hourly_count_plotly = st.container()
container_create_wordcloud = st.container()
container_create_word_frequency_figure = st.container()


with sidebar:
    st.title('Load sample data')
    if st.button('Click to load sample data'):
        file_name = 'sample_chat.txt'
        df = my_functions.extract_data_from_file_path(file_name)
        st.session_state.df = df
        st.success('Sample data loaded successfully!')
      
    
with container_setup:
    st.title('WhatsApp Chat Analysis')
    
    text = '''Hello! You can upload your group chat and view the analysis. 
    To learn how to download your WhatsApp group chat, please check out this _.'''
    url = "https://youtu.be/Dv5d7RKUyGY"
    st.markdown(f"{text.replace('_', f'[YouTube link]({url})')}", unsafe_allow_html=True)

        
with container_load_model:       
    if 'is_language_models_loaded' not in st.session_state:    
        # display a loading message while the model is loading
        with st.spinner("Loading language models..."):
            nlp_en, nlp_tr = my_functions.load_language_models()
            st.session_state.nlp_en = nlp_en
            st.session_state.nlp_tr = nlp_tr     
        # st.success("Language models are loaded.")
        st.session_state.is_language_models_loaded = True
    
    
with container_upload_file:
    uploaded_file = st.file_uploader("Upload the WhatsApp chat file (.txt file)", type=['txt'])
    text = '''<p style='font-size: small'><strong>*</strong> 
    I do not have access to the uploaded chats and they are not stored anywhere.</p>'''
    st.markdown(text, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Loading the data..."):
            df = my_functions.extract_data_from_streamlit_input(uploaded_file)
            st.session_state.df = df

        
with container_select_language:
    options = ["EN", "TR"]
    selected_option = st.selectbox("Select the language of the text", options)
    if selected_option == "EN":
        st.session_state.model = st.session_state.nlp_en
        st.session_state.clean_fn = my_functions.clean_punct_stop_space_lemma
    elif selected_option == "TR":
        st.session_state.model = st.session_state.nlp_tr
        st.session_state.clean_fn = my_functions.clean_punct_stop_space

        
with container_preprocess:
    if 'df' in st.session_state:
        df = st.session_state.df
        df = my_functions.add_datetime_column(df)
        df = my_functions.add_date_columns(df)
        df = my_functions.clean_and_count_words(df, st.session_state.clean_fn, st.session_state.model)
        st.session_state.df = df
        
        
with container_plot_message_count_vs_time_bar:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = my_functions.plot_message_count_vs_time_bar(st.session_state.df)
            st.plotly_chart(fig)
        
        
with container_plot_total_messages_by_sender:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = my_functions.plot_total_messages_by_sender(st.session_state.df)
            st.plotly_chart(fig)
        

with container_create_cumulative_count_bar_chart:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = my_functions.create_cumulative_count_bar_chart(st.session_state.df)
            st.plotly_chart(fig)


with container_plot_hourly_count_plotly:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = my_functions.plot_hourly_count_plotly(st.session_state.df)
            st.plotly_chart(fig)
        
                
with container_create_wordcloud:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            plt = my_functions.create_wordcloud(df)
            st.pyplot(plt)
            

with container_create_word_frequency_figure:
    if 'df' in st.session_state:
        fig = my_functions.create_word_frequency_figure(df, 5, 6)
        st.plotly_chart(fig)
        
    