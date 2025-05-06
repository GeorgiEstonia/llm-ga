"""
Flask API Server for LLM-GA Prompt Optimization

This server exposes endpoints to optimize prompts using genetic algorithms,
transform content using optimized prompts, and evaluate the quality of
transformed content.
"""

import os
import json
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import uuid
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Import GA adapter for the endpoints
import ga_adapter
from prompt_cache import prompt_cache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbose logging
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Remove file handler to prevent creating api_detailed_log.txt
        # logging.FileHandler("api_detailed_log.txt")  # Add file logging
    ]
)
logger = logging.getLogger("llm-ga-api")
logger.info("======== LLM-GA API SERVER STARTING ========")

# Initialize Flask app
app = Flask(__name__)

# In-memory storage for tasks
tasks = {}
tasks_lock = threading.Lock()  # Add thread lock for tasks dictionary

executor = ThreadPoolExecutor(max_workers=4)

# Available tone styles
TONE_STYLES = [
    "professional",
    "casual",
    "friendly",
    "formal",
    "technical",
    "persuasive",
    "enthusiastic",
    "youtube_thumbnails"
]
logger.info(f"Available tone styles: {TONE_STYLES}")

# Task cleanup thread
def periodic_cleanup():
    """Periodically clean up old optimization tasks"""
    while True:
        try:
            ga_adapter.clean_old_tasks(max_age_hours=24)
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
        time.sleep(3600)  # Run every hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup)
cleanup_thread.daemon = True
cleanup_thread.start()
logger.info("Cleanup thread started")

def run_optimization(task_id, tone_style, example_texts, neutral_examples, pop_size, generations):
    """
    Run the optimization process in the background using the GA adapter
    """
    try:
        logger.info(f"Starting real GA optimization for task {task_id} with tone_style={tone_style}")
        
        # Call the GA adapter's start_optimization function
        try:
            # This will launch the optimization process asynchronously
            result = ga_adapter.start_optimization(
                tone_style=tone_style,
                example_texts=example_texts,
                neutral_examples=neutral_examples,
                pop_size=pop_size,
                generations=generations,
                task_id=task_id  # Pass the task_id explicitly
            )
            
            logger.info(f"GA optimization started successfully for task {task_id}")
            
            # No need to monitor the task here - ga_adapter handles updating the task status
            
        except Exception as e:
            logger.error(f"Error starting GA optimization: {e}", exc_info=True)
            with tasks_lock:
                tasks[task_id] = {
                    "status": "failed",
                    "error": f"Failed to start optimization: {str(e)}",
                    "tone_style": tone_style
                }
    
    except Exception as e:
        logger.error(f"Unexpected error in run_optimization: {e}", exc_info=True)
        with tasks_lock:
            tasks[task_id] = {
                "status": "failed",
                "error": str(e)
            }

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running"""
    logger.info("Health check request received")
    response = {
        "status": "healthy", 
        "timestamp": time.time(),
        "cached_tones": len(prompt_cache.list_tones())
    }
    logger.info(f"Health check response: {json.dumps(response)}")
    return jsonify(response)

# List cached tone styles
@app.route("/api/tones", methods=["GET"])
def list_tones():
    """List all cached tone styles"""
    logger.info("Request received to list cached tone styles")
    tones = prompt_cache.list_tones()
    logger.info(f"Returning {len(tones)} cached tone styles")
    return jsonify({
        "tones": tones
    })

# Optimize prompt endpoint
@app.route("/api/optimize", methods=["POST"])
def optimize_prompt():
    """
    Start an optimization process for a tone style
    
    Request JSON:
    {
        "tone_style": "Professional and Informative",
        "example_texts": ["Example 1", "Example 2"],
        "neutral_examples": ["Neutral 1", "Neutral 2"],
        "pop_size": 7,           // Optional
        "generations": 3         // Optional
    }
    
    Response JSON:
    {
        "task_id": "uuid",
        "status": "preparing",
        "tone_style": "Professional and Informative"
    }
    """
    try:
        # Parse request
        data = request.json
        logger.info(f"Optimize prompt request received: {json.dumps(data)}")
        
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["tone_style", "example_texts"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        tone_style = data["tone_style"]
        example_texts = data["example_texts"]
        neutral_examples = data.get("neutral_examples", [])
        pop_size = data.get("pop_size", 20)
        generations = data.get("generations", 5)
        
        # Validate tone style
        if tone_style not in TONE_STYLES:
            logger.warning(f"Invalid tone style: {tone_style}")
            return jsonify({"error": f"Invalid tone style. Available options: {', '.join(TONE_STYLES)}"}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Start optimization in background
        executor.submit(
            run_optimization,
            task_id,
            tone_style,
            example_texts,
            neutral_examples,
            pop_size,
            generations
        )
        
        logger.info(f"Optimization task {task_id} started for tone style: {tone_style}")
        logger.info(f"Parameters: pop_size={pop_size}, generations={generations}")
        logger.info(f"Example counts: {len(example_texts)} examples, {len(neutral_examples)} neutral examples")
        
        # Return task ID
        return jsonify({
            "task_id": task_id,
            "status": "running",
            "tone_style": tone_style
        })
    
    except Exception as e:
        logger.error(f"Error in optimize_prompt: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Check optimization status
@app.route("/api/optimize/<task_id>", methods=["GET"])
def check_optimization(task_id):
    """
    Check the status of an optimization task
    
    Response JSON:
    {
        "task_id": "uuid",
        "status": "completed|failed|in_progress|not_found",
        "tone_style": "Professional and Informative",
        "result": {
            "prompt": "...",
            "fitness_score": 8.5,
            "feedback": "..."
        }
    }
    """
    logger.info(f"Checking status of optimization task: {task_id}")
    
    # First check with the GA adapter
    ga_status = ga_adapter.get_optimization_status(task_id)
    
    # If found in GA adapter, use that
    if ga_status.get("status") != "not_found":
        logger.info(f"Found task {task_id} in GA adapter with status: {ga_status.get('status')}")
        return jsonify(ga_status)
    
    # If not found in GA adapter, check our local tasks
    with tasks_lock:
        if task_id in tasks:
            logger.info(f"Found task {task_id} in local tasks with status: {tasks[task_id].get('status')}")
            task_data = tasks[task_id].copy()  # Make a copy to avoid holding the lock while serializing
            return jsonify(task_data)
    
    # If not found in either place, return not found
    logger.info("Task not found")
    return jsonify({"error": "Task not found"}), 404

# Transform text using optimized prompt
@app.route("/api/transform", methods=["POST"])
def transform_text():
    """
    Transform text using an optimized prompt
    
    Request JSON:
    {
        "tone_style": "Professional and Informative",
        "text": "Text to transform",
        "use_cached": true          // Optional
    }
    
    Response JSON:
    {
        "status": "success|error",
        "transformed_text": "Transformed text",
        "tone_style": "Professional and Informative",
        "prompt_used": "..."
    }
    """
    try:
        # Parse request
        data = request.json
        logger.info(f"Transform text request received with tone style: {data.get('tone_style', 'unknown')}, text length: {len(data.get('text', ''))}")
        
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["tone_style", "text"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        tone_style = data["tone_style"]
        text = data["text"]
        use_cached = data.get("use_cached", True)
        
        # Validate tone style
        if tone_style not in TONE_STYLES:
            logger.warning(f"Invalid tone style: {tone_style}")
            return jsonify({"error": f"Invalid tone style. Available options: {', '.join(TONE_STYLES)}"}), 400
        
        # Transform text
        result = ga_adapter.transform_text(
            tone_style=tone_style,
            text=text,
            use_cached=use_cached
        )
        
        if "error" in result:
            logger.warning(f"Error transforming text: {result['error']}")
            return jsonify({
                "status": "error",
                "error": result["error"]
            }), 400
        
        logger.info(f"Text transformation successful, response length: {len(result['transformedText'])}")
        
        # Return transformed text
        return jsonify({
            "transformed_text": result["transformedText"],
            "prompt_used": result.get("metadata", {}).get("promptUsed", "unknown"),
            "tone_style": tone_style
        })
    
    except Exception as e:
        logger.error(f"Error in transform_text: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Evaluate transformation
@app.route("/api/evaluate", methods=["POST"])
def evaluate_transformation():
    """
    Evaluate a text transformation
    
    Request JSON:
    {
        "original_text": "Original text",
        "transformed_text": "Transformed text",
        "tone_style": "Professional and Informative"
    }
    
    Response JSON:
    {
        "status": "success|error",
        "fitness_score": 8.5,
        "feedback": "..."
    }
    """
    try:
        # Parse request
        data = request.json
        logger.info(f"Evaluate transformation request received")
        
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["original_text", "transformed_text", "tone_style"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        original_text = data["original_text"]
        transformed_text = data["transformed_text"]
        tone_style = data["tone_style"]
        
        logger.info(f"Evaluating transformation for tone style: {tone_style}")
        
        # Evaluate transformation
        result = ga_adapter.evaluate_transformation(
            original_text=original_text,
            transformed_text=transformed_text,
            tone_style=tone_style
        )
        
        if "error" in result:
            logger.warning(f"Error evaluating transformation: {result['error']}")
            return jsonify({
                "status": "error",
                "error": result["error"]
            }), 400
        
        logger.info(f"Evaluation successful: score={result.get('score', 0)}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in evaluate_transformation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Clear cache endpoint
@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """
    Clear the prompt cache for a tone style or all tone styles
    
    Request JSON:
    {
        "tone_style": "Professional and Informative"   // Optional
    }
    
    Response JSON:
    {
        "status": "success",
        "message": "Cache cleared"
    }
    """
    try:
        data = request.json or {}
        tone_style = data.get("tone_style")
        
        if tone_style:
            logger.info(f"Clearing cache for tone style: {tone_style}")
            prompt_cache.clear(tone_style)
            return jsonify({
                "status": "success",
                "message": f"Cache cleared for tone style: {tone_style}"
            })
        else:
            logger.info("Clearing entire prompt cache")
            prompt_cache.clear()
            return jsonify({
                "status": "success",
                "message": "All cached prompts cleared"
            })
    
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Error handling
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Not found: {request.path}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Run the app if executed directly
if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting LLM-GA API server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
    logger.info("API server shutting down") 