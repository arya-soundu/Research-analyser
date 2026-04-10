import streamlit as st
from pypdf2 import PdfReader
st.title('Research Paper Analzser')

#Upload section
uploaded_file=st.file_uploader("Upload your Pdf",type='pdf')
if uploaded_file:
    #Extract text from pdf
    reader=PdfReader(uploaded_file)
    text=""
    for page in reader.pages:
        text+=page.extract_text()

    st.write("Text successfully extracted!! Length:",len(text))
    st.text_area("Preview",text[:500]) #Shows the first 500 characters of the text
    