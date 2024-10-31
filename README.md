# pdfRAGchatBot

pdfRAGchatBot is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about the content of uploaded PDF documents and receive accurate responses.
This project leverages Groq API for language processing and is designed to be an easy-to-use solution for document-based Q&A tasks.

## Features

- **PDF Upload**: Easily upload PDF files for analysis.
- **Interactive Q&A**: Ask questions related to the document and receive relevant answers.
- **Session Management**: Unique session IDs for maintaining chat history across app restarts.
- **User Authentication**: Login functionality with user details stored in JSON format. (Currently in progress)

## Getting Started

### Prerequisites

- Python 3.x
- Necessary libraries specified in `requirements.txt`

### Installation

1. Clone the repository
2. Navigate to the project directory and install required packages :-
   `pip install -r requirements.txt`
3. Run the application using streamlit :-
   `streamlit run main.py`
