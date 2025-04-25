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
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llm-ga-api")

# Initialize Flask app
app = Flask(__name__)

# In-memory storage for tasks
tasks = {}

executor = ThreadPoolExecutor(max_workers=4)

# Available tone styles
TONE_STYLES = [
    "professional",
    "casual",
    "friendly",
    "formal",
    "technical",
    "persuasive",
    "enthusiastic"
]

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

def run_optimization(task_id, tone_style, example_texts, neutral_examples, pop_size, generations):
    """
    Run the optimization process in the background
    This is a placeholder for the actual genetic algorithm implementation
    """
    try:
        # Simulate optimization process
        # This would be replaced with the actual genetic algorithm
        import time
        import random
        
        # Simulate work
        time.sleep(5)
        
        # Generate a mock optimized prompt
        optimized_prompt = f"Transform the following text to sound more {tone_style}. " + \
                          "Maintain the original meaning but adjust the tone, vocabulary, " + \
                          f"and sentence structure to reflect a {tone_style} style."
        
        # Update task with results
        tasks[task_id] = {
            "status": "completed",
            "prompt": optimized_prompt,
            "stats": {
                "generations": generations,
                "population_size": pop_size,
                "final_fitness": random.random() * 0.5 + 0.5,  # Random score between 0.5 and 1.0
                "convergence": True
            }
        }
    except Exception as e:
        tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "cached_tones": len(prompt_cache.list_tones())
    })

# List cached tone styles
@app.route("/api/tones", methods=["GET"])
def list_tones():
    """List all cached tone styles"""
    return jsonify({
        "tones": prompt_cache.list_tones()
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
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["tone_style", "example_texts"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        tone_style = data["tone_style"]
        example_texts = data["example_texts"]
        neutral_examples = data.get("neutral_examples", [])
        pop_size = data.get("pop_size", 20)
        generations = data.get("generations", 5)
        
        # Validate tone style
        if tone_style not in TONE_STYLES:
            return jsonify({"error": f"Invalid tone style. Available options: {', '.join(TONE_STYLES)}"}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task
        tasks[task_id] = {
            "status": "running",
            "parameters": {
                "tone_style": tone_style,
                "example_count": len(example_texts),
                "neutral_count": len(neutral_examples),
                "pop_size": pop_size,
                "generations": generations
            }
        }
        
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
        
        # Return task ID
        return jsonify({
            "task_id": task_id,
            "status": "running",
            "tone_style": tone_style
        })
    
    except Exception as e:
        logger.error(f"Error starting optimization: {e}", exc_info=True)
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
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    
    return jsonify(tasks[task_id])

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
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["tone_style", "text"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        tone_style = data["tone_style"]
        text = data["text"]
        use_cached = data.get("use_cached", True)
        
        # Transform text
        result = ga_adapter.transform_text(
            tone_style=tone_style,
            text=text,
            use_cached=use_cached
        )
        
        # Check for error
        if result.get("status") == "error":
            return jsonify(result), 400
        
        # Return transformed text
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error transforming text: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

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
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["original_text", "transformed_text", "tone_style"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        original_text = data["original_text"]
        transformed_text = data["transformed_text"]
        tone_style = data["tone_style"]
        
        # Evaluate transformation
        result = ga_adapter.evaluate_transformation(
            original_text=original_text,
            transformed_text=transformed_text,
            tone_style=tone_style
        )
        
        # Return evaluation
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error evaluating transformation: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

# Clear cache endpoint
@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """
    Clear the prompt cache
    
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
        # Parse request
        data = request.json or {}
        
        # Extract parameters
        tone_style = data.get("tone_style")
        
        # Clear cache
        prompt_cache.clear(tone_style)
        
        # Return success
        return jsonify({
            "status": "success",
            "message": f"Cache {'for ' + tone_style if tone_style else 'completely'} cleared"
        })
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Run the app if executed directly
if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting LLM-GA API server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug) 