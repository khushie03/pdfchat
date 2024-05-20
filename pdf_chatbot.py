import streamlit as st
from PyPDF2 import PdfReader
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch
from streamlit_option_menu import option_menu
import os
from serpapi import GoogleSearch
from tutorial import generate_gemini_content , extract_transcript_details
st.set_page_config(layout="wide")

st.sidebar.title('📚🔍 LLM Chat App: Your PDF Companion')
st.sidebar.markdown("Welcome to **LLM Chat App**: Your ultimate companion for interacting with PDF documents. Upload your PDFs and explore their contents with ease. Whether you need answers, summaries, or insights, this app has got you covered! Made with ❤️ by Khushi.")


def get_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_answer(question, context):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    with torch.no_grad():
        output = model.generate(input_ids)
        
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def generate_questions(context):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    inputs = tokenizer([context], max_length=1024, return_tensors="pt", truncation=True)
    question_ids = model.generate(inputs.input_ids, max_length=50, num_return_sequences=4, length_penalty=2.0, num_beams=4, early_stopping=True)
    questions = [tokenizer.decode(q_id, skip_special_tokens=True) for q_id in question_ids]
    return questions

def fetch_links(query):
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": "your_serp_apikey"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def chat_with_pdf():
    st.write("CHAT WITH PDF")
    pdf = st.file_uploader("UPLOAD YOUR PDF YOU WANT TO CHAT WITH", type="pdf")
    question = st.text_input("Enter your question")
    if pdf:
        chat_option = option_menu(
            menu_title=None,
            options=["Summarize", "Generate Answers and Questions", "Generate Links"],
            icons=["text", "info-circle", "link"],
            orientation="horizontal",
        )
        
        if chat_option == "Summarize":
            text = get_text_from_pdf(pdf)
            summary = summarize_text(text)
            st.write("Summary of the PDF:")
            st.write(summary)
        
        elif chat_option == "Generate Answers and Questions":
            text = get_text_from_pdf(pdf)
            
            
            if question:
                
                st.write("Top Answer:")
                answer = generate_answer(question, text)
                st.write(answer)
            generate_questions_flag = st.checkbox("Generate questions from the text")
            if generate_questions_flag:
                questions = generate_questions(text)
                st.write("Generated Questions:")
                for q in questions:
                    st.write(q)
        
        elif chat_option == "Generate Links":
            query = question
            if query:
                results = fetch_links(query)
                if 'search_metadata' in results and 'youtube_url' in results['search_metadata']:
                    youtube_url = results['search_metadata']['youtube_url']
                    st.write("YouTube URL:")
                    st.write(f"<a href='{youtube_url}'>Visit Youtube</a>", unsafe_allow_html=True)
                else:
                    st.write("YouTube URL not found in search metadata.")
                
                if 'ads_results' in results:
                    st.write("Other Links Section")
                    ads_results = results.get('ads_results', [])
                    for ad in ads_results:
                        link = ad.get('link', '')
                        title = ad.get('title', '')
                        st.write(f"Title: {title}")
                        st.write(f"<a href='{link}'>Visit Link</a>", unsafe_allow_html=True)

                search_results = results.get('organic_results', [])
                for result in search_results:
                    link = result.get('link', '')
                    title = result.get('title', '')
                    st.write(f"Title: {title}")
                    st.write(f"<a href='{link}'>Visit Link</a>", unsafe_allow_html=True)



            else:
                st.write("PLEASE UPLOAD YOUR PDF")

def scholar_section():
    query = st.text_input("Write the topic or subject you want to search")
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": "your_Serp_api_key"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    for result in organic_results:
        st.markdown(f"**Title:** {result['title']}")
        st.markdown(f"**Link:** {result['link']}")
        st.markdown(f"**Snippet:** {result['snippet']}")

        st.markdown("---")

def main():

    section = option_menu(
        menu_title= "Your Favourite Tutor",
        options=["Chat with PDF", "Your Scholar Friend" , "Youtube Notes"],
        icons=["info-circle", "link"],
        orientation="horizontal",
    )
    
    if section == "Chat with PDF":
        chat_with_pdf()
    elif section == "Your Scholar Friend":
        scholar_section()
    
    elif section == "Youtube Notes":
        prompt="""You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 1000 words. Please provide the summary of the text given here:  """
        st.write("Convert Your Youtube Lecture into detailed notes")
        youtube_link = st.text_input("Enter YouTube Video Link:")
        if youtube_link:
            video_id = youtube_link.split("=")[1]
            print(video_id)
        if st.button("Get Detailed Notes"):
            transcript_text=extract_transcript_details(youtube_link)
            if transcript_text:
                summary=generate_gemini_content(transcript_text,prompt)
                st.markdown("Notes from the Video Link")
                st.write(summary)

if __name__ == "__main__":
    main()
