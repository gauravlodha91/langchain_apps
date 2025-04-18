import logging
from typing import List, Dict, Any
import cohere

logger = logging.getLogger(__name__)

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
                preamble=f"You are an AI assistant that answers questions based on YouTube video transcripts. Do Not add additional information from Your side. If The provided question does not have context then simple return No-Context Found. Use the following information from YouTube videos to answer the user's question: {context}",
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