import os
import sys
import time
from datetime import datetime
import json
import requests
import streamlit as st

# Add this helper function at the top of the file after imports
def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return the ID if already in correct format."""
    if 'youtube.com' in url_or_id or 'youtu.be' in url_or_id:
        # Handle youtube.com URLs
        if 'v=' in url_or_id:
            return url_or_id.split('v=')[1].split('&')[0]
        # Handle youtu.be URLs
        elif 'youtu.be' in url_or_id:
            return url_or_id.split('/')[-1].split('?')[0]
    return url_or_id.strip()

# Configure page settings
st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

# API endpoint (change in production)
API_ENDPOINT = "http://localhost:8000"

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "notification" not in st.session_state:
    st.session_state.notification = None

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    
    if st.button("Apply Chunking Settings"):
        try:
            response = requests.post(
                f"{API_ENDPOINT}/update-settings",
                json={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            )
            if response.status_code == 200:
                st.success("Settings applied!")
            else:
                st.error(f"Failed to apply settings: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Add button to retrieve all chunks as JSON
    if st.button("Export All Chunks as JSON"):
        try:
            response = requests.get(f"{API_ENDPOINT}/chunks")
            if response.status_code == 200:
                chunks_data = response.json()
                st.json(chunks_data)
                st.success(f"Exported {chunks_data['count']} chunks as JSON.")
            else:
                st.error(f"Failed to retrieve chunks: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Show collection stats with refresh button
    st.subheader("Collection Statistics")
    if st.button("Refresh Stats"):
        st.session_state.stats = None  # Force refresh

    if "stats" not in st.session_state or st.session_state.stats is None:
        try:
            response = requests.get(f"{API_ENDPOINT}/stats")
            if response.status_code == 200:
                st.session_state.stats = response.json()
            else:
                st.session_state.stats = {}
                st.error("Could not retrieve collection stats")
        except Exception as e:
            st.session_state.stats = {}
            st.error(f"Error: {str(e)}")

    stats = st.session_state.get("stats", {})
    st.write(f"Total chunks: {stats.get('total_chunks', 0)}")
    st.write(f"Number of videos: {stats.get('estimated_videos', stats.get('total_videos', 0))}")

    # Display source scores for last message if available
    if "chat_history" in st.session_state and st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        if last_message["role"] == "assistant" and "sources" in last_message:
            st.subheader("Search Results Scores")
            sources = last_message["sources"]
            for i, source in enumerate(sources):
                st.write(f"**Result {i+1}:**")
                st.write(f"Video: {source['video_name']}")
                st.write(f"Author: {source['author']}")
                st.write(f"Score: {source['score']}")
                st.write("---")

# Main tabs
tab1, tab2 = st.tabs(["Chat with YouTube Videos", "Add YouTube Videos"])

# Tab 1: Chat interface
with tab1:
    st.title("Chat with YouTube Videos")
    
    # Create a container for the chat interface
    chat_interface = st.container()
    
    with chat_interface:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                st.markdown(f"**[{source['video_name']}]({source['timestamp_link']})** by {source['author']}")
                                st.write(f"Snippet: {source['snippet']}")
                                st.write(f"Score: {source['score']}")
                                st.write("---")
        
        # Chat input (always at the bottom)
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
                    try:
                        # Query the API
                        response = requests.post(
                            f"{API_ENDPOINT}/query",
                            json={"query": user_query, "n_results": 5}
                        )
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            
                            # Display response
                            st.write(response_data["response"])
                            
                            # Display sources in an expander
                            if response_data["sources"]:
                                with st.expander("Sources"):
                                    for source in response_data["sources"]:
                                        st.markdown(f"**[{source['video_name']}]({source['timestamp_link']})** by {source['author']}")
                                        st.write(f"Snippet: {source['snippet']}")
                                        st.write(f"Score: {source['score']}")
                                        st.write("---")
                        else:
                            error_msg = f"Error from API: {response.text}"
                            st.error(error_msg)
                            response_data = {"response": error_msg, "sources": []}
                    except Exception as e:
                        error_msg = f"Error connecting to API: {str(e)}"
                        st.error(error_msg)
                        response_data = {"response": error_msg, "sources": []}
            
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
            # Parse and clean input into list of video IDs
            video_ids = []
            invalid_inputs = []
            
            for line in video_input.strip().split('\n'):
                if line.strip():
                    video_id = extract_video_id(line.strip())
                    if video_id:
                        video_ids.append(video_id)
                    else:
                        invalid_inputs.append(line)
            
            if invalid_inputs:
                st.error(f"Invalid video URLs/IDs found: {', '.join(invalid_inputs)}")
                
            if video_ids:
                try:
                    with st.spinner("Starting video processing..."):
                        response = requests.post(
                            f"{API_ENDPOINT}/process-videos",
                            json={
                                "video_ids": video_ids,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap
                            }
                        )
                        
                        if response.status_code == 200:
                            st.success(f"Started processing {len(video_ids)} videos")
                            st.info("You can monitor the processing status below")
                            
                            # Show processed video IDs
                            with st.expander("Processed Video IDs", expanded=True):
                                for vid_id in video_ids:
                                    st.code(vid_id)
                        else:
                            st.error(f"Failed to start processing: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
                    st.info("Please check if the API server is running")

    # Only show All Videos Status section as requested
    st.markdown("---")
    st.subheader("Video Processing Status")
    
    # Display function for status badges
    def display_status_badge(status):
        """Display a formatted status badge with icon and message."""
        if status == "completed":
            st.success("✅ COMPLETED")
        elif status == "processing":
            st.warning("⏳ PROCESSING")
        elif status == "started":
            st.info("🆕 STARTED")
        elif status == "failed":
            st.error("❌ FAILED")
        elif status == "not_found":
            st.warning("⚠️ NOT FOUND")
        else:
            st.info(f"ℹ️ {status.upper()}")
    
    # Display all video statuses
    if st.button("Refresh All Video Status"):
        try:
            response = requests.get(f"{API_ENDPOINT}/all-video-status")
            if response.status_code == 200:
                all_status = response.json()
                
                if all_status:
                    for video_id, status_data in all_status.items():
                        with st.expander(f"Video ID: {video_id}", expanded=True):
                            display_status_badge(status_data["status"])
                            if status_data["error"]:
                                st.error(f"Error: {status_data['error']}")
                                # Show notification if transcript is missing
                                if (
                                    status_data["status"] == "failed"
                                    and (
                                        "no transcript" in status_data["error"].lower()
                                        or "could not be processed" in status_data["error"].lower()
                                    )
                                ):
                                    st.session_state.notification = f"Video {video_id}: {status_data['error']}"
                else:
                    st.info("No videos in processing queue")
            else:
                st.error(f"Failed to retrieve status: {response.text}")
        except Exception as e:
            st.error(f"Error checking status: {str(e)}")

    # Show notification if set
    if st.session_state.notification:
        st.warning(st.session_state.notification)
        time.sleep(5)
        st.session_state.notification = None
