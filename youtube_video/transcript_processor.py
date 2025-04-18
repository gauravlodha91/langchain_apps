import re
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import YouTube

logger = logging.getLogger(__name__)

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
            # Try English first
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                logger.info("Transcript fetched in English for video ID: %s", video_id)
                return transcript_list
            except Exception:
                # Try any available language
                transcript_info = YouTubeTranscriptApi.list_transcripts(video_id)
                for transcript in transcript_info:
                    if transcript.is_translatable:
                        transcript_list = transcript.fetch()
                        if translate_to_english:
                            for item in transcript_list:
                                item["text"] = self.translator.translate(item["text"], dest="en").text
                        logger.info("Transcript fetched and translated for video ID: %s", video_id)
                        return transcript_list
                logger.warning("No translatable transcript found for video ID: %s", video_id)
                return []
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
            ## check w1
            
            processed_chunk = {
                "chunk_id": f"{video_metadata['video_id']}-{i}",
                "text": chunk_text,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "video_id": video_metadata['video_id'],
                **video_metadata  # Include all video metadata
            }
            processed_chunks.append(processed_chunk)
        
        logger.info("Processed %d chunks for video ID: %s", len(processed_chunks), video_metadata.get("video_id", "Unknown"))
        return processed_chunks
    
    def process_video(self, collection, video_id_or_url):
        """
        Process a single video and index its chunks.
        This is a non-async method to avoid event loop issues in background tasks.
        """
        try:
            # Extract the video ID if a URL was provided
            video_id = self.extract_video_id(video_id_or_url) if '/' in str(video_id_or_url) else video_id_or_url
            
            # Get video metadata
            video_metadata = self.get_video_metadata(video_id)
            
            # Get transcript
            transcript_list = self.get_transcript(video_id)
            
            # Process the transcript into chunks
            processed_chunks = self.process_transcript(transcript_list, video_metadata)
            
            # Index the chunks
            logger.info(f"Indexing {len(processed_chunks)} chunks into ChromaDB for video {video_id}")
            result = self.index_chunks(collection, processed_chunks)
            
            # Return processed chunks count for success tracking
            return {"success": result, "chunks": len(processed_chunks)}
        except Exception as e:
            logger.error(f"Error processing video {video_id_or_url}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_video_info(self, video_id):
        """
        Get video information using a synchronous method.
        Adjust this based on how you're currently getting video info.
        """
        try:
            # This is a placeholder - replace with your actual implementation
            # that doesn't use async/await
            import requests
            
            # Example using YouTube Data API
            api_key = os.getenv("YOUTUBE_API_KEY")
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
            
            response = requests.get(url)
            data = response.json()
            
            if 'items' in data and data['items']:
                snippet = data['items'][0]['snippet']
                return {
                    "title": snippet.get("title", "Unknown"),
                    "author": snippet.get("channelTitle", "Unknown")
                }
            return None
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def index_chunks(self, collection, processed_chunks: List[Dict[str, Any]]):
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
    
    def process_and_index_videos(self, collection, video_ids: List[str], progress_callback=None) -> Dict[str, Any]:
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
                
                result = self.process_video(collection, video_id)
                
                if result["success"]:
                    results["success"].append({
                        "video_id": video_id,
                        "chunks": result["chunks"]
                    })
                else:
                    results["failed"].append({
                        "video_id": video_id,
                        "reason": result.get("error", "Failed to index chunks")
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