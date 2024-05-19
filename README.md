# pdfchat
LLM Chat App
Overview
The LLM Chat App is a web application built using Streamlit that allows users to interact with PDF documents, 
search for scholarly articles, and generate notes from YouTube videos. The application integrates various APIs 
and libraries to provide a seamless user experience for accessing and summarizing educational content.
Features
- Chat with PDF: Users can upload PDF documents and interact with them by summarizing text, generating 
answers and questions, and fetching related links.
- Your Scholar Friend: This feature enables users to search for scholarly articles using Google Scholar and 
view relevant titles, links, and snippets.
- Youtube Notes: Users can convert YouTube lectures into detailed notes by providing the video link, which is 
then summarized using Google's Generative AI model.
How to Use
1. Installation:
 - Clone this repository to your local machine.
 - Install the required dependencies by running `pip install -r requirements.txt`.
2. Running the App:
 - Navigate to the project directory in your terminal.
 - Run the Streamlit app using the command `streamlit run app.py`.
 - The app will open in your default web browser.
3. Usage:
 - Choose from the available options in the sidebar to access different features of the app.
 - Follow the prompts and input fields to interact with PDFs, search for scholarly articles, and generate notes 
from YouTube videos.
Technologies Used
- Streamlit: Frontend framework for building interactive web applications with Python.
- PyPDF2: Library for reading and manipulating PDF files in Python.
- Hugging Face Transformers: Library for natural language processing tasks, including text summarization and 
question answering.
- PyTorch: Deep learning framework used for machine learning tasks in the application.
- SerpApi: API for performing web searches and extracting search results.
- YouTube Transcript API: API for fetching transcripts from YouTube videos.
- dotenv: Library for managing environment variables in Python.
