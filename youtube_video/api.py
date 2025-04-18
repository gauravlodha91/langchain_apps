import os
import logging
import re
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

def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extracts the YouTube video ID from a URL or returns the ID if already valid.
    Returns None if extraction fails.
    """
    url_or_id = url_or_id.strip()
    # Direct ID
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    # youtube.com/watch?v=...
    match = re.search(r"(?:v=)([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    # youtu.be/...
    match = re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    return None

# API routes
@app.get("/")
async def root():
    return {"message": "YouTube RAG Chatbot API is running"}

@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed videos."""
    stats = get_collection_stats(collection)
    # Ensure stats contains 'estimated_videos'
    if "estimated_videos" not in stats:
        chunks = get_all_chunks(collection)
        # DEBUG: Print first 3 chunks to check their structure
        logger.info(f"First 3 chunks: {chunks[:3]}")
        unique_videos = set(chunk.get("video_id") for chunk in chunks if "video_id" in chunk)
        stats["estimated_videos"] = len(unique_videos)
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

    # Validate and extract video IDs
    valid_video_ids = []
    for raw_id in request.video_ids:
        vid = extract_video_id(raw_id)
        if vid:
            valid_video_ids.append(vid)
            video_status[vid] = {"status": "started", "error": None}
        else:
            video_status[raw_id] = {"status": "failed", "error": "Invalid video ID or URL"}

    if not valid_video_ids:
        raise HTTPException(status_code=400, detail="No valid video IDs found.")

    # Create processor with requested settings
    custom_processor = YouTubeTranscriptProcessor(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )

    # Background task for processing videos
    def process_videos_background(video_ids):
        for video_id in video_ids:
            try:
                video_status[video_id]["status"] = "processing"
                result = custom_processor.process_video(collection, video_id)
                if isinstance(result, dict) and result.get("success"):
                    if result.get("chunks", 0) > 0:
                        video_status[video_id]["status"] = "completed"
                        video_status[video_id]["error"] = None
                    else:
                        video_status[video_id]["status"] = "failed"
                        video_status[video_id]["error"] = "No transcript found or transcript could not be processed."
                else:
                    video_status[video_id]["status"] = "failed"
                    video_status[video_id]["error"] = result.get("error", "Failed to process video") if isinstance(result, dict) else "Failed to process video"
            except Exception as e:
                video_status[video_id]["status"] = "failed"
                video_status[video_id]["error"] = str(e)
                logger.error(f"Error processing video {video_id}: {str(e)}")

    # Start processing in background
    background_tasks.add_task(process_videos_background, valid_video_ids)

    return {
        "message": f"Processing {len(valid_video_ids)} videos in the background",
        "video_ids": valid_video_ids
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