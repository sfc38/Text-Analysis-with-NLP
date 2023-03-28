import streamlit as st

st.write("Hello! This is Fatih.")
    
text = '''You can upload your group chat and view the analysis. To learn how to download your WhatsApp group chat, please check out this _.'''
url = "https://youtu.be/Dv5d7RKUyGY"
st.markdown(f"{text.replace('_', f'[YouTube link]({url})')}", unsafe_allow_html=True)
    
st.write('''I do not have access to the uploaded chats and they are not stored anywhere.''')

st.write('''Go back to the app and try the analyzer.''')
