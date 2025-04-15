import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import re
import asyncio  # Add asyncio for handling async operations
import logging  # Add logging module

import streamlit as st
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
import chromadb
from chromadb.utils import embedding_functions
import requests
from pytube import YouTube  # Import YouTube class from pytube

from dotenv import load_dotenv
load_dotenv()


# Configure logging
log_filename = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

logger.info("Application started.")

# Set up environment variables and API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Replace with your actual API key or set in Streamlit secrets

# Initialize Cohere client for embeddings
co = cohere.Client(COHERE_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.Client()
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0"  # Using Cohere's embedding model
)

# Ensure the collection exists
try:
    collection = chroma_client.get_collection(
        name="youtube_transcripts",
        embedding_function=cohere_ef
    )
except:
    collection = chroma_client.create_collection(
        name="youtube_transcripts",
        embedding_function=cohere_ef
    )

class YouTubeTranscriptProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        logger.info("Initializing YouTubeTranscriptProcessor with chunk_size=%d and chunk_overlap=%d", chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.translator = Translator()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def extract_video_id(self, url_or_id: str) -> str:
        logger.info("Extracting video ID from input: %s", url_or_id)
        # YouTube ID pattern
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, url_or_id)
        if match:
            video_id = match.group(1)
        elif len(url_or_id) == 11:  # It's likely already an ID
            video_id = url_or_id
        else:
            raise ValueError(f"Could not extract video ID from {url_or_id}")
        logger.info("Extracted video ID: %s", video_id)
        return video_id
    
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        logger.info("Fetching metadata for video ID: %s", video_id)
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            metadata = {
                "video_id": video_id,
                "video_name": yt.title,
                "author": yt.author,
                "video_created_day": yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else "Unknown",
                "video_processed_date": datetime.now().strftime("%Y-%m-%d"),
                "description": yt.description,
                "view_count": yt.views,
                "length": yt.length
            }
        except Exception as e:
            logger.error("Error getting metadata for video %s: %s", video_id, e)
            metadata = {
                "video_id": video_id,
                "video_name": "Unknown",
                "author": "Unknown",
                "video_created_day": "Unknown",
                "video_processed_date": datetime.now().strftime("%Y-%m-%d"),
            }
        logger.info("Retrieved metadata: %s", metadata)
        return metadata
    
    def get_transcript(self, video_id: str, translate_to_english: bool = True) -> List[Dict[str, Any]]:
        logger.info("Fetching transcript for video ID: %s", video_id)
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Translate if needed and not in English
            if translate_to_english:
                # Check if transcript is already in English
                sample_text = " ".join([item["text"] for item in transcript_list[:5]])
                
                # Use asyncio to handle the async detect method
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                detected_lang = loop.run_until_complete(self.translator.detect(sample_text))
                
                if detected_lang.lang != 'en':  # Correctly access the 'lang' attribute
                    # Translate each segment
                    for i, segment in enumerate(transcript_list):
                        try:
                            translated = self.translator.translate(segment["text"], dest='en')
                            transcript_list[i]["text"] = translated.text
                        except Exception as e:
                            logger.error("Translation error for segment %d: %s", i, e)
                            # Keep original text if translation fails
                            pass
            
            logger.info("Transcript fetched successfully for video ID: %s", video_id)
            return transcript_list
        except Exception as e:
            logger.error("Error getting transcript for video %s: %s", video_id, e)
            return []
    
    def process_transcript(self, transcript_list: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Processing transcript for video ID: %s", video_metadata.get("video_id", "Unknown"))
        if not transcript_list:
            return []
        
        # Combine transcript segments into a single text
        full_transcript = " ".join([item["text"] for item in transcript_list])
        
        # Map transcript timestamps
        timestamp_map = {}
        current_position = 0
        for item in transcript_list:
            text_length = len(item["text"])
            end_position = current_position + text_length
            timestamp_map[(current_position, end_position)] = {
                "start": item["start"],
                "duration": item.get("duration", 0)
            }
            current_position = end_position + 1  # +1 for the space
        
        # Split into chunks
        chunks = self.text_splitter.create_documents([full_transcript])
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            chunk_start_char = full_transcript.find(chunk_text)
            chunk_end_char = chunk_start_char + len(chunk_text)
            
            # Find the best matching timestamp
            start_time = 0
            end_time = 0
            for (start_pos, end_pos), time_data in timestamp_map.items():
                if start_pos <= chunk_start_char and end_pos >= chunk_start_char:
                    start_time = time_data["start"]
                if start_pos <= chunk_end_char and end_pos >= chunk_end_char:
                    end_time = time_data["start"] + time_data.get("duration", 0)
                    break
            
            processed_chunk = {
                "chunk_id": f"{video_metadata['video_id']}-{i}",
                "text": chunk_text,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                **video_metadata  # Include all video metadata
            }
            processed_chunks.append(processed_chunk)
        
        logger.info("Processed %d chunks for video ID: %s", len(processed_chunks), video_metadata.get("video_id", "Unknown"))
        return processed_chunks
    
    def process_video(self, video_id_or_url: str, translate_to_english: bool = True) -> List[Dict[str, Any]]:
        """Process a YouTube video: get transcript, metadata, and create chunks."""
        video_id = self.extract_video_id(video_id_or_url)
        video_metadata = self.get_video_metadata(video_id)
        transcript_list = self.get_transcript(video_id, translate_to_english)
        processed_chunks = self.process_transcript(transcript_list, video_metadata)
        return processed_chunks
    
    def index_chunks(self, processed_chunks: List[Dict[str, Any]]):
        logger.info("Indexing %d chunks into ChromaDB", len(processed_chunks))
        if not processed_chunks:
            return False
        
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        
        for chunk in processed_chunks:
            chunk_text = chunk.pop("text")
            chunk_id = chunk.pop("chunk_id")
            
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            chunk_metadatas.append(chunk)
        
        try:
            collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            logger.info("Chunks indexed successfully.")
            return True
        except Exception as e:
            logger.error("Error indexing chunks: %s", e)
            return False
    
    def process_and_index_videos(self, video_ids: List[str], progress_callback=None) -> Dict[str, Any]:
        logger.info("Starting processing and indexing for %d videos", len(video_ids))
        results = {
            "success": [],
            "failed": []
        }
        
        total_videos = len(video_ids)
        for i, video_id in enumerate(video_ids):
            try:
                if progress_callback:
                    progress_callback(i / total_videos, f"Processing video {i+1}/{total_videos}: {video_id}")
                
                chunks = self.process_video(video_id)
                success = self.index_chunks(chunks)
                
                if success:
                    results["success"].append({
                        "video_id": video_id,
                        "chunks": len(chunks)
                    })
                else:
                    results["failed"].append({
                        "video_id": video_id,
                        "reason": "Failed to index chunks"
                    })
            except Exception as e:
                logger.error("Error processing video %s: %s", video_id, e)
                if progress_callback:
                    progress_callback(i / total_videos, f"Error with video {video_id}: {str(e)}")
                
                results["failed"].append({
                    "video_id": video_id,
                    "reason": str(e)
                })
        
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        logger.info("Processing and indexing completed. Success: %d, Failed: %d", len(results["success"]), len(results["failed"]))
        return results


class RAGChatbot:
    def __init__(self, cohere_api_key, collection):
        logger.info("Initializing RAGChatbot with Cohere API key and collection")
        self.co = cohere.Client(cohere_api_key)
        self.collection = collection
    
    def search_similar_chunks(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        logger.info("Searching for similar chunks for query: %s", query)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        if results and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    "id": doc_id,
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        logger.info("Found %d similar chunks for query: %s", len(formatted_results), query)
        return formatted_results
    
    def format_youtube_timestamp_link(self, video_id: str, seconds: float) -> str:
        """Format a YouTube link with timestamp."""
        return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}"
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Generating response for query: %s", query)
        if not search_results:
            return {
                "response": "I couldn't find any relevant information from the indexed YouTube videos.",
                "sources": []
            }
        
        # Limit to the first 3 results
        search_results = search_results[:3]
        
        # Prepare context from search results
        context = ""
        sources = []
        
        for result in search_results:
            metadata = result["metadata"]
            context += f"\n\nFrom video '{metadata['video_name']}' by {metadata['author']}:\n{result['text']}"
            
            # Format source with YouTube timestamp
            timestamp_link = self.format_youtube_timestamp_link(
                metadata['video_id'], 
                metadata['start_time']
            )
            
            sources.append({
                "video_id": metadata['video_id'],
                "video_name": metadata['video_name'],
                "author": metadata['author'],
                "timestamp_link": timestamp_link,
                "start_time": metadata['start_time'],
                "snippet": result['text'][:150] + "..." if len(result['text']) > 150 else result['text'],
                "score": result.get("distance", None)  # Include the score
            })
        
        # Generate response with Cohere
        try:
            chat_response = self.co.chat(
                message=query,
                preamble=f"You are an AI assistant that answers questions based on YouTube video transcripts. Do Not add additional information from Your side. If The provided question does not have context then simple return NO Context Found. Use the following information from YouTube videos to answer the user's question: {context}",
                model="command-r",
                temperature=0.7,
            )
            
            response_data = {
                "response": chat_response.text,
                "sources": sources
            }
        except Exception as e:
            logger.error("Error generating response: %s", e)
            response_data = {
                "response": f"I encountered an error while generating the response: {str(e)}",
                "sources": sources
            }
        
        logger.info("Response generated successfully for query: %s", query)
        return response_data


# Add method to retrieve all chunks from the vector database
def get_all_chunks_as_json():
    logger.info("Retrieving all chunks from the vector database")
    try:
        all_chunks = collection.get(include=["metadatas", "documents"])
        chunks_json = [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata
            }
            for chunk_id, document, metadata in zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"])
        ]
        logger.info("Retrieved %d chunks from the vector database", len(chunks_json))
        return chunks_json
    except Exception as e:
        logger.error("Error retrieving chunks: %s", e)
        return []


# Streamlit Application
def main():
    logger.info("Starting Streamlit application")
    st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
    
    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "processor" not in st.session_state:
        st.session_state.processor = YouTubeTranscriptProcessor()
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(COHERE_API_KEY, collection)
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
        
        if st.button("Apply Chunking Settings"):
            st.session_state.processor = YouTubeTranscriptProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.success("Settings applied!")
        
        # Add button to retrieve all chunks as JSON
        if st.button("Export All Chunks as JSON"):
            chunks_json = get_all_chunks_as_json()
            if chunks_json:
                st.json(chunks_json)
                st.success(f"Exported {len(chunks_json)} chunks as JSON.")
            else:
                st.error("Failed to retrieve chunks. Check logs for details.")
        
        # Show collection stats
        st.subheader("Collection Statistics")
        try:
            count = collection.count()
            st.write(f"Total chunks: {count}")
            
            if count > 0:
                sample = collection.peek(10)
                video_ids = set()
                for metadata in sample["metadatas"]:
                    video_ids.add(metadata["video_id"])
                st.write(f"Number of videos: {len(video_ids)} (estimated)")
        except Exception as e:
            logger.error("Error getting collection stats: %s", e)
            st.error(f"Error getting collection stats: {e}")
    
    # with st.sidebar:
    #     st.title("Search Results Scores")
        
    #     if "chat_history" in st.session_state and st.session_state.chat_history:
    #         last_message = st.session_state.chat_history[-1]
    #         if last_message["role"] == "assistant" and "sources" in last_message:
    #             sources = last_message["sources"]
    #             for i, source in enumerate(sources):
    #                 st.write(f"**Result {i+1}:**")
    #                 st.write(f"Video: {source['video_name']}")
    #                 st.write(f"Author: {source['author']}")
    #                 st.write(f"Score: {source['score']}")
    #                 st.write("---")
    
    # Main tabs
    tab1, tab2 = st.tabs(["Chat with YouTube Videos", "Add YouTube Videos"])
    
    # Tab 1: Chat interface
    with tab1:
        st.title("Chat with YouTube Videos")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                # st.markdown(f"**[{source['video_name']}]({source['timestamp_link']})** by {source['author']}")
                                st.write(f"Snippet: {source['snippet']}")
                                st.write(f"Score: {source['score']}")
                                st.write("---")
        
        # Chat input
        user_query = st.chat_input("Ask a question about the YouTube content...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Get response from chatbot
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Search for similar chunks
                    search_results = st.session_state.chatbot.search_similar_chunks(user_query, n_results=5)
                    
                    # Generate response
                    response_data = st.session_state.chatbot.generate_response(user_query, search_results)
                    
                    # Display response
                    st.write(response_data["response"])
                    
                    # Display sources in an expander
                    if response_data["sources"]:
                        with st.expander("Sources"):
                            for source in response_data["sources"]:
                                # st.markdown(f"**[{source['video_name']}]({source['timestamp_link']})** by {source['author']}")
                                st.write(f"Snippet: {source['snippet']}")
                                st.write(f"Score: {source['score']}")
                                st.write("---")
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data["response"],
                "sources": response_data["sources"]
            })
    
    # Tab 2: Add YouTube Videos
    with tab2:
        st.title("Add YouTube Videos")
        
        st.write("""
        Add YouTube videos to the knowledge base by entering video IDs or URLs below.
        You can enter one video per line.
        """)
        
        # Input for video IDs or URLs
        video_input = st.text_area(
            "Enter YouTube Video IDs or URLs (one per line):",
            height=150
        )
        
        # Process button
        if st.button("Process Videos"):
            if not video_input.strip():
                st.error("Please enter at least one video ID or URL.")
            else:
                # Parse input into list of video IDs
                video_ids = [line.strip() for line in video_input.split("\n") if line.strip()]
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Define progress callback
                def update_progress(progress, status):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                # Process videos
                with st.spinner("Processing videos..."):
                    results = st.session_state.processor.process_and_index_videos(
                        video_ids,
                        progress_callback=update_progress
                    )
                
                # Display results
                st.success(f"Processing complete! Successfully processed {len(results['success'])} videos.")
                
                if results["success"]:
                    st.subheader("Successfully processed videos:")
                    for video in results["success"]:
                        st.write(f"- {video['video_id']} ({video['chunks']} chunks)")
                
                if results["failed"]:
                    st.subheader("Failed videos:")
                    for video in results["failed"]:
                        st.write(f"- {video['video_id']}: {video['reason']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Application encountered an error: %s", str(e))