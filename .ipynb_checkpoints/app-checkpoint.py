import streamlit as st
import spacy
import spacy.cli


# check if the en_core_web_sm model is already installed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # download the en_core_web_sm model from spaCy
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


st.header("Remove stopwords and punctuation")

def clean_text(text, nlp):
    doc = nlp(text) 
    # exclude punctuation and stop words
    words = [token.text for token in doc if not token.is_punct and not token.is_stop] 
    return words

# options = ["Option 1", "Option 2"]
# selected_option = st.selectbox("Select an option", options)
# st.write("You selected:", selected_option)

text_input = st.text_area("Enter some text", height=100)
st.write("You entered:")
if text_input is not None:
    st.write(text_input)

    words = clean_text(text_input, nlp)

    st.write("Output:")
    st.write(words)