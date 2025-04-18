import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from database import initialize_database, get_all_chunks, get_collection_stats
from transcript_processor import YouTubeTranscriptProcessor
from rag_chatbot import RAGChatbot
from utils import setup_logging, load_api_keys

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logging()

# Load API keys
api_keys = load_api_keys()
cohere_api_key = api_keys["cohere_api_key"]

# Initialize database and services
collection = initialize_database(cohere_api_key)
processor = YouTubeTranscriptProcessor()
chatbot = RAGChatbot(cohere_api_key, collection)

# Add status tracking
video_status = {}  # Dictionary to store video processing status

# Create FastAPI app
app = FastAPI(title="YouTube RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests and responses
class VideoRequest(BaseModel):
    video_ids: List[str]
    chunk_size: int = 500
    chunk_overlap: int = 50

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3

class ChunkSettingsRequest(BaseModel):
    chunk_size: int
    chunk_overlap: int

class VideoStatus(BaseModel):
    video_id: str
    status: str
    error: Optional[str] = None

# API routes
@app.get("/")
async def root():
    return {"message": "YouTube RAG Chatbot API is running"}

@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed videos."""
    stats = get_collection_stats(collection)
    return stats

@app.get("/chunks")
async def get_chunks():
    """Get all indexed chunks."""
    chunks = get_all_chunks(collection)
    return {"chunks": chunks, "count": len(chunks)}

@app.post("/process-videos")
async def process_videos(request: VideoRequest, background_tasks: BackgroundTasks):
    """Process and index YouTube videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="No video IDs provided")
    
    # Initialize status for each video
    for video_id in request.video_ids:
        video_status[video_id] = {"status": "started", "error": None}
    
    # Create processor with requested settings
    custom_processor = YouTubeTranscriptProcessor(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    
    # Simplified background task function
    def process_videos_background(video_ids):
        for video_id in video_ids:
            try:
                video_status[video_id]["status"] = "processing"
                # Process video without using nested event loops
                success = custom_processor.process_video(collection, video_id)
                if success:
                    video_status[video_id]["status"] = "completed"
                else:
                    video_status[video_id]["status"] = "failed"
                    video_status[video_id]["error"] = "Failed to process video"
            except Exception as e:
                video_status[video_id]["status"] = "failed"
                video_status[video_id]["error"] = str(e)
                logger.error(f"Error processing video {video_id}: {str(e)}")
    
    # Start processing in background
    background_tasks.add_task(process_videos_background, request.video_ids)
    
    return {
        "message": f"Processing {len(request.video_ids)} videos in the background",
        "video_ids": request.video_ids
    }
@app.post("/query")
async def query(request: QueryRequest):
    """Query the chatbot with a question."""
    search_results = chatbot.search_similar_chunks(request.query, n_results=request.n_results)
    response = chatbot.generate_response(request.query, search_results)
    return response

@app.post("/update-settings")
async def update_settings(request: ChunkSettingsRequest):
    """Update chunking settings."""
    global processor
    processor = YouTubeTranscriptProcessor(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    return {"message": "Settings updated successfully"}

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    """Get the processing status of a specific video."""
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    return VideoStatus(
        video_id=video_id,
        status=video_status[video_id]["status"],
        error=video_status[video_id]["error"]
    )

@app.get("/all-video-status")
async def get_all_video_status():
    """Get the processing status of all videos."""
    return {
        video_id: VideoStatus(
            video_id=video_id,
            status=status["status"],
            error=status["error"]
        ) for video_id, status in video_status.items()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)