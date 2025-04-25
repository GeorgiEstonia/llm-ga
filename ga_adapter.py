"""
GA Adapter Module

Provides a simplified interface to the GA functionality for the API.
Acts as an adapter between the API endpoints and the GA implementation.
"""

import os
import time
import logging
import threading
import tempfile
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import GA components
from main import Individual, PromptGA, load_pairs, PAIRS_CSV_IN
from ga_llm_steps import run_stylize_chain, run_fitness_chain
from generate_population import generate_initial_population
from prompt_cache import prompt_cache

# Configure logging
logger = logging.getLogger("ga-adapter")

# Dictionary to track GA optimization processes
optimization_tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

def create_example_pairs(
    neutral_texts: List[str],
    stylized_texts: List[str],
    temp_dir: Optional[str] = None
) -> str:
    """
    Create a CSV file with neutral-stylized text pairs for GA training
    
    Args:
        neutral_texts: List of neutral/source texts
        stylized_texts: List of target stylized texts (must match neutral_texts length)
        temp_dir: Optional temporary directory for the CSV file
        
    Returns:
        Path to the created CSV file
    """
    if len(neutral_texts) != len(stylized_texts):
        raise ValueError("neutral_texts and stylized_texts must have the same length")
    
    if len(neutral_texts) < 2:
        raise ValueError("At least 2 example pairs are required for GA optimization")
    
    # Create a temporary directory if not provided
    if not temp_dir:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    csv_path = os.path.join(temp_dir, f"pairs_{uuid.uuid4().hex}.csv")
    
    # Create CSV file with headers
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("stylised_text,neutralised_text\n")
        for stylized, neutral in zip(stylized_texts, neutral_texts):
            # Escape quotes and newlines for CSV format
            stylized_escaped = stylized.replace('"', '""').replace("\n", " ")
            neutral_escaped = neutral.replace('"', '""').replace("\n", " ")
            f.write(f'"{stylized_escaped}","{neutral_escaped}"\n')
    
    logger.info(f"Created example pairs CSV at {csv_path} with {len(neutral_texts)} pairs")
    return csv_path

def start_optimization(
    tone_style: str, 
    example_texts: List[str],
    neutral_examples: List[str],
    pop_size: int = 7,
    generations: int = 3
) -> str:
    """
    Start an asynchronous optimization process for a tone style
    
    Args:
        tone_style: The tone style to optimize for
        example_texts: Examples of text in the desired tone
        neutral_examples: Neutral versions of the example texts
        pop_size: Population size for GA
        generations: Number of generations to run
        
    Returns:
        Task ID for tracking the optimization
    """
    # Check if already in cache
    cached = prompt_cache.get(tone_style)
    if cached:
        logger.info(f"Found cached prompt for '{tone_style}', returning cached version")
        task_id = str(uuid.uuid4())
        with tasks_lock:
            optimization_tasks[task_id] = {
                "tone_style": tone_style,
                "status": "completed",
                "timestamp": time.time(),
                "result": {
                    "prompt": cached["prompt"],
                    "fitness_score": cached.get("fitness_score", 0),
                    "from_cache": True
                }
            }
        return task_id
    
    # Create a new task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task data
    with tasks_lock:
        optimization_tasks[task_id] = {
            "tone_style": tone_style,
            "status": "preparing",
            "timestamp": time.time(),
            "result": None
        }
    
    # Start the optimization in a separate thread
    thread = threading.Thread(
        target=_run_optimization,
        args=(task_id, tone_style, example_texts, neutral_examples, pop_size, generations)
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started optimization task {task_id} for tone style '{tone_style}'")
    return task_id

def _run_optimization(
    task_id: str, 
    tone_style: str, 
    example_texts: List[str],
    neutral_examples: List[str],
    pop_size: int,
    generations: int
):
    """
    Run the GA optimization process (to be called in a separate thread)
    
    Args:
        task_id: The task ID for tracking
        tone_style: The tone style to optimize for
        example_texts: Examples of text in the desired tone
        neutral_examples: Neutral versions of the example texts
        pop_size: Population size for GA
        generations: Number of generations to run
    """
    try:
        # Update task status
        with tasks_lock:
            if task_id in optimization_tasks:
                optimization_tasks[task_id]["status"] = "creating_examples"
        
        # Create a temporary directory for this optimization
        temp_dir = tempfile.mkdtemp()
        
        # Create example pairs CSV
        pairs_csv = create_example_pairs(neutral_examples, example_texts, temp_dir)
        
        # Update task status
        with tasks_lock:
            if task_id in optimization_tasks:
                optimization_tasks[task_id]["status"] = "optimizing"
                optimization_tasks[task_id]["pairs_csv"] = pairs_csv
        
        # Load pairs from CSV
        pairs = load_pairs(Path(pairs_csv))
        
        # Initialize GA with the pairs
        ga = PromptGA(
            text_category=tone_style,
            pairs=pairs,
            pop_size=pop_size,
            generations=generations,
            survival_ratio=0.5,  # Keep top 50%
            num_eval_pairs=2  # Use up to 2 pairs for evaluation
        )
        
        # Run the GA
        ga.run()
        
        # Get the best individual
        best_individual = None
        best_fitness = -1
        
        for ind in ga.individuals:
            if ind.fitness_score > best_fitness:
                best_fitness = ind.fitness_score
                best_individual = ind
        
        # Check if we found a good solution
        if best_individual and best_individual.fitness_score > 0:
            # Save to cache
            prompt_data = {
                "prompt": best_individual.prompt,
                "fitness_score": best_individual.fitness_score,
                "feedback": best_individual.feedback,
                "generation": best_individual.generation,
                "timestamp": time.time()
            }
            prompt_cache.save(tone_style, prompt_data)
            
            # Update task status
            with tasks_lock:
                if task_id in optimization_tasks:
                    optimization_tasks[task_id]["status"] = "completed"
                    optimization_tasks[task_id]["result"] = {
                        "prompt": best_individual.prompt,
                        "fitness_score": best_individual.fitness_score,
                        "feedback": best_individual.feedback,
                        "generation": best_individual.generation
                    }
            
            logger.info(f"Optimization task {task_id} completed successfully with fitness {best_fitness}")
        else:
            # Update task status
            with tasks_lock:
                if task_id in optimization_tasks:
                    optimization_tasks[task_id]["status"] = "failed"
                    optimization_tasks[task_id]["error"] = "No suitable prompt found"
            
            logger.warning(f"Optimization task {task_id} failed: No suitable prompt found")
    
    except Exception as e:
        logger.error(f"Optimization task {task_id} failed with error: {e}", exc_info=True)
        # Update task status
        with tasks_lock:
            if task_id in optimization_tasks:
                optimization_tasks[task_id]["status"] = "failed"
                optimization_tasks[task_id]["error"] = str(e)
    
    finally:
        # Clean up temporary files
        try:
            if "pairs_csv" in optimization_tasks.get(task_id, {}):
                csv_path = optimization_tasks[task_id]["pairs_csv"]
                if os.path.exists(csv_path):
                    os.unlink(csv_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def get_optimization_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of an optimization task
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Dict with task status information
    """
    with tasks_lock:
        if task_id in optimization_tasks:
            # Return a copy to avoid thread safety issues
            return dict(optimization_tasks[task_id])
        else:
            return {"status": "not_found", "error": "Task ID not found"}

def transform_text(
    tone_style: str,
    text: str,
    use_cached: bool = True
) -> Dict[str, Any]:
    """
    Transform a text using an optimized prompt
    
    Args:
        tone_style: The tone style to use
        text: The text to transform
        use_cached: Whether to use cached prompts
        
    Returns:
        Dict with the transformed text and metadata
    """
    # Check if we have a cached prompt
    cached_prompt = None
    if use_cached:
        cached_data = prompt_cache.get(tone_style)
        if cached_data and "prompt" in cached_data:
            cached_prompt = cached_data["prompt"]
    
    if not cached_prompt:
        return {
            "status": "error",
            "error": f"No optimized prompt found for tone style '{tone_style}'",
            "transformed_text": None
        }
    
    try:
        # Prepare the payload for the stylizer
        payload = {
            "style_prompt": cached_prompt,
            "neutralized_text_to_transform": text,
            # These are dummy values just to satisfy the stylizer API,
            # they're not actually used since we're providing the prompt directly
            "neutralized_text_example": "Example neutral text.",
            "stylized_text_example": "Example stylized text."
        }
        
        # Transform the text using the stylizer
        transformed_text = run_stylize_chain(payload)
        
        if not transformed_text:
            return {
                "status": "error",
                "error": "Stylizer returned empty text",
                "transformed_text": None
            }
        
        return {
            "status": "success",
            "transformed_text": transformed_text,
            "tone_style": tone_style,
            "prompt_used": cached_prompt[:100] + "..." if len(cached_prompt) > 100 else cached_prompt
        }
    
    except Exception as e:
        logger.error(f"Error transforming text with tone '{tone_style}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "transformed_text": None
        }

def evaluate_transformation(
    original_text: str,
    transformed_text: str,
    tone_style: str
) -> Dict[str, Any]:
    """
    Evaluate the quality of a transformation
    
    Args:
        original_text: The original text
        transformed_text: The transformed text
        tone_style: The target tone style
        
    Returns:
        Dict with evaluation results
    """
    try:
        # Prepare the payload for the fitness evaluator
        payload = {
            "tone_style": tone_style,
            "neutral_text": original_text,
            "stylized_text": transformed_text
        }
        
        # Evaluate the transformation
        evaluation = run_fitness_chain(payload)
        
        if not evaluation or "fitness_score" not in evaluation:
            return {
                "status": "error",
                "error": "Fitness evaluation failed",
                "evaluation": None
            }
        
        return {
            "status": "success",
            "fitness_score": evaluation["fitness_score"],
            "feedback": evaluation["feedback"]
        }
    
    except Exception as e:
        logger.error(f"Error evaluating transformation: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "evaluation": None
        }

def list_cached_tones() -> Dict[str, Dict[str, Any]]:
    """
    List all cached tone styles
    
    Returns:
        Dict mapping tone styles to metadata
    """
    return prompt_cache.list_tones()

def clean_old_tasks(max_age_hours: int = 24):
    """
    Clean up old optimization tasks
    
    Args:
        max_age_hours: Maximum age in hours for tasks to keep
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    with tasks_lock:
        to_remove = []
        for task_id, task in optimization_tasks.items():
            if now - task.get("timestamp", 0) > max_age_seconds:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del optimization_tasks[task_id]
    
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old optimization tasks") 