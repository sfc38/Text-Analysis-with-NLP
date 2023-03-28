import streamlit as st
from my_functions import *
import pandas as pd
import io
import re

container_setup = st.container()
container_load_model = st.container()
sidebar = st.sidebar
container_preprocess = st.container()
container_plot_message_count_vs_time_bar = st.container()
container_plot_total_messages_by_sender = st.container()
container_create_cumulative_count_bar_chart = st.container()
container_plot_hourly_count_plotly = st.container()
container_create_wordcloud = st.container()
container_create_word_frequency_figure = st.container()


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
        df = extract_data_from_file_path(file_name)
        st.session_state.df = df      
    
    
with container_preprocess:
    if 'df' in st.session_state:
        with st.spinner("Processing data..."):
            df = st.session_state.df
            df = add_datetime_column(df)
            df = add_date_columns(df)
            df = clean_and_count_words(df, st.session_state.clean_fn, st.session_state.model)
            st.session_state.df = df
        
        
with container_plot_message_count_vs_time_bar:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = plot_message_count_vs_time_bar(st.session_state.df)
            st.plotly_chart(fig)
        
        
with container_plot_total_messages_by_sender:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = plot_total_messages_by_sender(st.session_state.df)
            st.plotly_chart(fig)
        

with container_create_cumulative_count_bar_chart:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = create_cumulative_count_bar_chart(st.session_state.df)
            st.plotly_chart(fig)


with container_plot_hourly_count_plotly:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = plot_hourly_count_plotly(st.session_state.df)
            st.plotly_chart(fig)
        
                
with container_create_wordcloud:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            plt = create_wordcloud(st.session_state.df)
            st.pyplot(plt)
            

with container_create_word_frequency_figure:
    if 'df' in st.session_state:
        with st.spinner("Loading the plot..."):
            fig = create_word_frequency_figure(st.session_state.df, 5, 6)
            st.plotly_chart(fig)        
    