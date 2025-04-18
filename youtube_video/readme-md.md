# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with content from YouTube videos. The application extracts, processes, and indexes YouTube video transcripts, then uses the Cohere API to generate responses based on relevant transcript chunks.

## Features

- Extract and process YouTube video transcripts
- Automatic translation of non-English transcripts
- Chunking of transcripts with customizable parameters
- Vector storage using ChromaDB
- RAG-based responses using Cohere API
- Streamlit frontend with chat interface
- FastAPI backend for scalability and separation of concerns

## Architecture

The project follows a clean architecture with separation between:

1. **Backend Services**:
   - YouTube transcript processing
   - Vector database management
   - RAG-based chatbot logic

2. **API Layer**:
   - RESTful endpoints for frontend interaction

3. **Frontend**:
   - Streamlit-based user interface

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Cohere API key
- Internet connection for YouTube API access

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd youtube-rag-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your Cohere API key.

### Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn api:app --reload
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```
   streamlit run app.py
   ```

3. Open the Streamlit application in your browser (typically at http://localhost:8501)

## Usage

### Adding YouTube Videos

1. Navigate to the "Add YouTube Videos" tab
2. Enter YouTube video URLs or video IDs (one per line)
3. Click "Process Videos" to extract and index the transcripts

### Chatting with YouTube Content

1. Navigate to the "Chat with YouTube Videos" tab
2. Enter your question in the chat input
3. View the response and the source videos with timestamps

### Configuration

- Adjust the chunk size and overlap in the sidebar
- Export all indexed chunks as JSON for debugging
- View collection statistics in the sidebar

## Project Structure

```
youtube-rag-chatbot/
├── api.py                # FastAPI backend service
├── app.py                # Streamlit frontend
├── database.py           # ChromaDB connection and operations
├── transcript_processor.py  # YouTube transcript processing
├── rag_chatbot.py        # RAG-based chatbot functionality
├── utils.py              # Utility functions
├── logging_config.py     # Logging configuration
├── .env                  # Environment variables (not in repo)
├── .env.example          # Environment variables template
├── requirements.txt      # Project dependencies
└── logs/                 # Log files directory
```

## License

[MIT License](LICENSE)
