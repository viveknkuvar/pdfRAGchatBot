# PDF_RAG_ChatBot

PDF_RAG_ChatBot is an application that uses Retrieval-Augmented Generation (RAG) architecture, enabling users to ask questions about the content of uploaded PDF documents and receive accurate responses. This project leverages the Groq API for language processing and is designed to be an easy-to-use solution for document-based Q&A tasks.

## Features

- **PDF Upload**: Easily upload PDF files for analysis.
- **Interactive Q&A**: Ask questions related to the document and receive relevant answers.
- **Session Management**: Unique session IDs for maintaining chat history across app restarts.
- **User Authentication**: Login functionality with user details stored in JSON format (currently in progress).

## Live Demo

You can try the application live at: [PDF RAG ChatBot](https://pdf-rag-chatbot-vnk.streamlit.app/)

## Getting Started

### Prerequisites

- Python 3.x
- Necessary libraries specified in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/viveknkuvar/pdfRAGchatBot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pdfRAGchatBot
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your own Groq and Nomic API keys in the api_keys.json file

## Usage

1. Run the application:
   ```bash
   streamlit run main.py
   ```
2. Open your browser and navigate to the local URL mentioned in the terminal to access the chatbot interface.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for any suggestions or improvements.

## Acknowledgements

- Groq API for language processing
- [@descentis](https://github.com/descentis) for their invaluable guidance during training, which helped me to complete and deploy this assignment.

## Contact

For any inquiries, please reach out to me at [viveknk777@gmail.com].
