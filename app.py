import streamlit as st
import spacy
import spacy.cli
import spacyturk

st.header("Remove stopwords and punctuation")

if 'is_language_models_loaded' not in st.session_state:    

    # display a loading message while the model is loading
    with st.spinner("Loading language models..."):

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
            
        st.session_state.nlp_en = nlp_en
        st.session_state.nlp_tr = nlp_tr
        
    # display the loaded model once it is done loading
    st.success("Language models are loaded.")
    
    st.session_state.is_language_models_loaded = True

def clean_text(text, nlp):
    doc = nlp(text) 
    # exclude punctuation and stop words
    words = [token.text for token in doc if not token.is_punct and not token.is_stop and not token.text.isspace()] 
    return words

options = ["EN", "TR"]
selected_option = st.selectbox("Select the language of the text", options)
if selected_option == "EN":
    st.session_state.model = st.session_state.nlp_en
elif selected_option == "TR":
    st.session_state.model = st.session_state.nlp_tr

text_input = st.text_area("Enter some text", height=100)
st.write("Input Text:")
if text_input is not None:
    st.write(text_input)

    words = clean_text(text_input, st.session_state.model)

    st.write("Cleaned text as list of words:")
    st.write(words)