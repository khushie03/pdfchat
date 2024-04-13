import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

st.sidebar.title('🤗💬 LLM Chat App')
st.sidebar.markdown("This allows you to upload the PDF and find answers from it")
st.sidebar.markdown("Made by Khushi")

def get_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.write("CHAT WITH PDF")
    pdf = st.file_uploader("UPLOAD YOUR PDF YOU WANT TO CHAT WITH ", type="pdf")
    question = st.text_input("Enter your question")

    if question:  
        if pdf:  
            text = get_text_from_pdf(pdf)
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")
            answer = qa_pipeline(question=question, context=text)
            st.write("Top Answer:")
            st.write(answer['answer'])
        else:
            st.write("PLEASE UPLOAD YOUR PDF")
    else:
        st.write("Please Write your question")

if __name__ == "__main__":
    main()
