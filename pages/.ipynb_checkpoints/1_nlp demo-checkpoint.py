import streamlit as st
import my_functions

st.header("Remove Stopwords, Punctuation, and Apply Lemmatization")
st.write("Note: Lemmatization is only available for the English language.")

if 'is_language_models_loaded' not in st.session_state:    
    
    # display a loading message while the model is loading
    with st.spinner("Loading language models..."):
        nlp_en, nlp_tr = my_functions.load_language_models()
        st.session_state.nlp_en = nlp_en
        st.session_state.nlp_tr = nlp_tr
        
    # display the loaded model once it is done loading
    st.success("Language models are loaded.")
    st.session_state.is_language_models_loaded = True

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
    
    if st.session_state.model == st.session_state.nlp_en:
        words = my_functions.clean_punct_stop_space_lemma(text_input, st.session_state.model)
    else:
        words = my_functions.clean_punct_stop_space(text_input, st.session_state.model)
        
    st.write("Cleaned text as list of words:")
    st.write(words)