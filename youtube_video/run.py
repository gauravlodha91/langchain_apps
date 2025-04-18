import os
import argparse
import subprocess
import time
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ["COHERE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment.")
        return False
    return True

def start_backend():
    """Start the FastAPI backend server."""
    print("Starting backend server...")
    return subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def start_frontend():
    """Start the Streamlit frontend."""
    print("Starting frontend application...")
    return subprocess.Popen(["streamlit", "run", "app.py"])

def handle_exit(backend_process, frontend_process):
    """Properly terminate processes on exit."""
    def signal_handler(sig, frame):
        print("\nShutting down...")
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            print("Frontend terminated.")
        
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            print("Backend terminated.")
        
        sys.exit(0)
    
    return signal_handler

def main():
    parser = argparse.ArgumentParser(description="Run YouTube RAG Chatbot")
    parser.add_argument("--backend-only", action="store_true", help="Run only the backend API")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the Streamlit frontend")
    args = parser.parse_args()
    
    # Check environment variables
    if not check_env_vars():
        return
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start processes based on args
        if not args.frontend_only:
            backend_process = start_backend()
            # Wait for backend to start
            time.sleep(2)
        
        if not args.backend_only:
            frontend_process = start_frontend()
        
        # Set up signal handlers for graceful termination
        signal.signal(signal.SIGINT, handle_exit(backend_process, frontend_process))
        signal.signal(signal.SIGTERM, handle_exit(backend_process, frontend_process))
        
        # Keep the script running
        print("\nApplication is running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print("Backend process terminated unexpectedly.")
                break
                
            if frontend_process and frontend_process.poll() is not None:
                print("Frontend process terminated unexpectedly.")
                break
            
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        # Clean up processes
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            print("Frontend terminated.")
        
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            print("Backend terminated.")

if __name__ == "__main__":
    main()