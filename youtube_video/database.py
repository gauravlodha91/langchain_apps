import os
import chromadb
from chromadb.utils import embedding_functions
import logging
import cohere

logger = logging.getLogger(__name__)

def initialize_database(cohere_api_key):
    """Initialize ChromaDB with Cohere embedding function."""
    logger.info("Initializing ChromaDB with Cohere embedding function")
    
    # Initialize ChromaDB
    chroma_client = chromadb.Client()
    
    # Set up Cohere embedding function
    cohere_ef = embedding_functions.CohereEmbeddingFunction(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0"
    )
    
    # Ensure the collection exists
    try:
        collection = chroma_client.get_collection(
            name="youtube_transcripts",
            embedding_function=cohere_ef
        )
        logger.info("Existing collection found: youtube_transcripts")
    except:
        collection = chroma_client.create_collection(
            name="youtube_transcripts",
            embedding_function=cohere_ef
        )
        logger.info("Created new collection: youtube_transcripts")
    
    return collection

def get_all_chunks(collection):
    """Retrieve all chunks from the vector database."""
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

def get_collection_stats(collection):
    """Get statistics about the collection."""
    try:
        count = collection.count()
        stats = {"total_chunks": count}
        
        if count > 0:
            sample = collection.peek(10)
            video_ids = set()
            for metadata in sample["metadatas"]:
                video_ids.add(metadata["video_id"])
            stats["estimated_videos"] = len(video_ids)
        
        return stats
    except Exception as e:
        logger.error("Error getting collection stats: %s", e)
        return {"error": str(e)}