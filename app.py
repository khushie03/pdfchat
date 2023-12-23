import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import os
import pickle

# Ensure the required libraries are installed
# pip install streamlit python-dotenv PyPDF2 scikit-learn transformers

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown("""
                This allows you to upload the pdf and find answers from it
                """)
    st.markdown("Made by Khushi")

def get_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_bert_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model(input_ids)
    word_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return word_embeddings

def main():
    st.write("CHAT WITH PDF")
    pdf = st.file_uploader("UPLOAD YOUR PDF YOU WANT TO CHAT WITH ", type="pdf")
    question = st.text_input("Enter your question")

    if pdf is not None:
        text = get_text_from_pdf(pdf)

        # Example: BERT embeddings
        # word_embeddings = get_bert_embeddings(text)

        # TF-IDF similarity
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])

        query_vector = tfidf_vectorizer.transform([question])
        similarity_scores = tfidf_matrix.dot(query_vector.T).toarray().flatten()
        top_indices = similarity_scores.argsort()[-3:][::-1]

        st.write("Top Answers:")
        for index in top_indices:
            answer = text.split('\n')[index]
            st.write(answer)

        store_name = pdf.name[:-4]
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump((tfidf_vectorizer, tfidf_matrix), f)

    else:
        st.write("PLEASE UPLOAD YOUR PDF")

if __name__ == "__main__":
    main()
