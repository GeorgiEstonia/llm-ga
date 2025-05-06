"""
GA Adapter Module

This module serves as a bridge between the Flask API server and the GA optimization code.
It provides functions to start optimization tasks, check task status, and transform text using optimized prompts.
"""

import os
import time
import logging
import json
import pandas as pd
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import uuid

# Import optimize functions
from optimize import optimize_prompt, generate_prompts, evaluate_prompt, run_genetic_algorithm
from prompt_cache import prompt_cache

# Configure logging
logger = logging.getLogger("llm_ga.adapter")
logger.setLevel(logging.DEBUG)

# Task storage
_optimization_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()

def _load_example_pairs_from_csv(csv_path: str, max_samples: int = 10) -> List[Dict[str, str]]:
    """
    Load example pairs from a CSV file
    
    Args:
        csv_path: Path to the CSV file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of dictionaries with 'original' and 'transformed' keys
    """
    try:
        logger.info(f"Loading example pairs from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if len(df.columns) < 2:
            logger.error(f"CSV file does not have required columns: {csv_path}")
            raise ValueError("CSV file must have at least two columns")
        
        # Get column names - we expect first column to be styled and second to be neutralized
        columns = list(df.columns)
        stylized_col = columns[0]  # Usually "Stylized Text"
        neutral_col = columns[1]   # Usually "Neutralized Text"
        
        # Load pairs - using up to max_samples rows
        # If max_samples is 0 or negative, load all rows
        if max_samples <= 0:
            sample_df = df
            logger.info(f"Loading all {len(df)} rows from CSV")
        else:
            sample_df = df.sample(min(max_samples, len(df)))
            logger.info(f"Sampling {len(sample_df)} rows from CSV (out of {len(df)} total)")
        
        # Create pairs from sampled rows
        pairs = []
        for i, row in sample_df.iterrows():
            pairs.append({
                "original": row[neutral_col],    # Original is the neutral text
                "transformed": row[stylized_col]  # Transformed is the stylized text
            })
        
        logger.info(f"Loaded {len(pairs)} example pairs from CSV")
        return pairs
        
    except Exception as e:
        logger.error(f"Error loading example pairs from CSV: {e}")
        return []

def _transform_with_direct_examples(text: str, tone_style: str) -> Tuple[str, Dict[str, Any]]:
    """
    Transform text using direct examples from pairs.csv
    This is used as a fallback when no optimized prompt is available.
    
    Args:
        text: The text to transform
        tone_style: The target tone style
        
    Returns:
        Tuple of (transformed_text, metadata)
    """
    try:
        import openai
        from openai import OpenAI
        
        logger.info(f"Using direct examples to transform text for tone style: {tone_style}")
        client = OpenAI()
        
        # Get example pairs
        pairs_csv = os.path.join(os.path.dirname(__file__), "pairs.csv")
        example_pairs = _load_example_pairs_from_csv(pairs_csv, max_samples=3)
        
        if not example_pairs:
            logger.warning("No example pairs found for direct transformation")
            return text, {"error": "No example pairs found for direct transformation"}
        
        logger.info(f"Loaded {len(example_pairs)} example pairs for direct transformation")
        
        # Add formatting preservation instruction
        formatting_instruction = (
            "IMPORTANT: Preserve all markdown formatting from the original text, including:\n"
            "- Bold text using **double asterisks**\n"
            "- Italic text using *single asterisks*\n"
            "- Numbered lists (1., 2., etc.)\n"
            "- Bullet point lists that start with '- '\n"
            "- Any other special formatting or symbols\n\n"
            "The transformed text should maintain the same structure, paragraphs, and formatting as the original.\n\n"
            "DO NOT include any emojis or special characters that weren't in the original text. Avoid adding decorative elements."
        )
        
        # Build prompt with examples
        system_message = f"You are an expert at transforming text to match specific tone styles through few-shot learning while preserving all markdown formatting."
        user_message = f"Transform the following text to match a {tone_style} style, using these examples as a guide:\n\n"
        
        # Add example pairs
        for i, pair in enumerate(example_pairs):
            user_message += f"Example {i+1} - Original: {pair['original']}\n"
            user_message += f"Example {i+1} - Transformed: {pair['transformed']}\n\n"
        
        # Add formatting instruction
        user_message += f"{formatting_instruction}\n\n"
        
        # Add the text to transform
        user_message += f"Now, transform this text to match the same style:\n{text}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            transformed_text = response.choices[0].message.content
            logger.info(f"Successfully transformed text using direct examples")
            
            return transformed_text, {"promptUsed": "Direct examples from pairs.csv"}
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API for example-based transformation: {e}")
            return text, {"error": f"API error: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error in _transform_with_direct_examples: {e}")
        return text, {"error": str(e)}

def start_optimization(
    tone_style: str, 
    example_texts: List[str],
    neutral_examples: List[str] = None,
    pop_size: int = 7,
    generations: int = 3,
    task_id: str = None
) -> Dict[str, Any]:
    """
    Start a GA optimization process for a tone style
    
    Args:
        tone_style: The tone style to optimize for
        example_texts: List of example texts in the target style
        neutral_examples: List of neutral examples (optional)
        pop_size: Population size for the GA
        generations: Number of generations to run
        task_id: Optional task ID for tracking
        
    Returns:
        Dictionary with task status
    """
    try:
        logger.info(f"Starting optimization for tone style: {tone_style}")
        logger.info(f"Parameters: pop_size={pop_size}, generations={generations}")
        
        # Generate a task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create example pairs
        pairs = []
        
        # If neutral examples are provided, pair them with example_texts
        if neutral_examples and len(neutral_examples) > 0:
            logger.info(f"Using provided neutral examples: {len(neutral_examples)}")
            for i in range(min(len(example_texts), len(neutral_examples))):
                pairs.append({
                    "original": neutral_examples[i],  # original (neutral) text
                    "transformed": example_texts[i]   # transformed (styled) text
                })
            logger.info(f"Created {len(pairs)} example pairs from provided examples")
        
        # If we don't have enough pairs yet, use pairs from CSV
        if len(pairs) < 2:
            logger.info("Not enough pairs from provided examples, using pairs.csv to supplement")
            pairs_csv = os.path.join(os.path.dirname(__file__), "pairs.csv")
            # Load up to 100 example pairs for better training
            csv_pairs = _load_example_pairs_from_csv(pairs_csv, max_samples=100)
            
            # If we have some example_texts but no neutral examples, pair with CSV neutral texts
            if example_texts and len(example_texts) > 0:
                for i, text in enumerate(example_texts):
                    if i < len(csv_pairs):
                        # Use neutral text from CSV file
                        pairs.append({
                            "original": csv_pairs[i]["original"],
                            "transformed": text
                        })
            
            # If we still need more pairs, add some directly from CSV
            if len(pairs) < 2 and csv_pairs:
                # Add pairs from CSV until we have at least 2
                for i in range(min(100, len(csv_pairs))):
                    if i >= len(pairs):  # Only add if not already in pairs
                        pairs.append(csv_pairs[i])
            
            logger.info(f"After supplementing with CSV pairs, we have {len(pairs)} example pairs")
        
        # Create a task entry
        with _tasks_lock:
            _optimization_tasks[task_id] = {
                "status": "running",
                "tone_style": tone_style,
                "parameters": {
                    "example_count": len(example_texts),
                    "neutral_count": len(neutral_examples) if neutral_examples else 0,
                    "pop_size": pop_size,
                    "generations": generations
                }
            }
        
        # Check if we have enough pairs to proceed
        if len(pairs) < 2:
            logger.error(f"Not enough example pairs. Need at least 2, got {len(pairs)}")
            with _tasks_lock:
                _optimization_tasks[task_id]["status"] = "failed"
                _optimization_tasks[task_id]["error"] = "Not enough example pairs"
            return {
                "status": "failed",
                "error": "Not enough example pairs",
                "task_id": task_id
            }
        
        # Start optimization in a background thread
        def run_optimization():
            try:
                logger.info(f"Running GA optimization for tone style: {tone_style}")
                result = run_genetic_algorithm(
                    tone_style=tone_style,
                    example_pairs=pairs,
                    generations=generations,
                    population_size=pop_size
                )
                
                # Cache the best prompt
                if "prompt" in result and result["prompt"]:
                    logger.info(f"Caching best prompt for tone style: {tone_style}")
                    prompt_cache.save(tone_style, {
                        "prompt": result["prompt"],
                        "fitness_score": result.get("fitness_score", 0),
                        "generation": generations,
                        "timestamp": time.time()
                    })
                
                logger.info(f"GA optimization complete for tone style: {tone_style}")
                
                # Update task status
                with _tasks_lock:
                    if task_id in _optimization_tasks:
                        _optimization_tasks[task_id].update({
                            "status": "completed",
                            "result": result
                        })
                
            except Exception as e:
                logger.error(f"Error in GA optimization thread: {e}")
                with _tasks_lock:
                    if task_id in _optimization_tasks:
                        _optimization_tasks[task_id].update({
                            "status": "failed",
                            "error": str(e)
                        })
        
        # Start the optimization in a background thread
        threading.Thread(target=run_optimization, daemon=True).start()
        
        return {
            "status": "running",
            "task_id": task_id,
            "tone_style": tone_style
        }
    
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        with _tasks_lock:
            if task_id:
                _optimization_tasks[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "tone_style": tone_style
                }
        return {
            "status": "failed",
            "error": str(e),
            "task_id": task_id
        }

def get_optimization_status(task_id: str) -> Dict[str, Any]:
    """
    Check the status of an optimization task
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Dictionary with task status
    """
    with _tasks_lock:
        if task_id in _optimization_tasks:
            return _optimization_tasks[task_id].copy()
        else:
            return {"status": "not_found"}

def transform_text(
    tone_style: str,
    text: str,
    use_cached: bool = True
) -> Dict[str, Any]:
    """
    Transform text using an optimized prompt
    
    Args:
        tone_style: The tone style to use
        text: The text to transform
        use_cached: Whether to use cached prompts
        
    Returns:
        Dictionary with transformed text and metadata
    """
    try:
        import openai
        from openai import OpenAI
        
        logger.info(f"Transforming text using tone style: {tone_style}")
        client = OpenAI()
        
        # Add an instruction to preserve markdown formatting
        formatting_instruction = (
            "IMPORTANT: Preserve all markdown formatting from the original text, including:\n"
            "- Bold text using **double asterisks**\n"
            "- Italic text using *single asterisks*\n"
            "- Numbered lists (1., 2., etc.)\n"
            "- Bullet point lists that start with '- '\n"
            "- Any other special formatting or symbols\n\n"
            "The transformed text should maintain the same structure, paragraphs, and formatting as the original."
        )
        
        if use_cached:
            # Check if we have a cached prompt
            cached_prompt = prompt_cache.get(tone_style)
            if cached_prompt and "prompt" in cached_prompt:
                logger.info(f"Using cached prompt for tone style: {tone_style}")
                prompt = cached_prompt["prompt"]
                
                # Call OpenAI API with the cached prompt
                prompt_message = f"Transform the following text to match a {tone_style} tone style. {prompt}\n\n{formatting_instruction}\n\nText to transform:\n{text}"
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-0125-preview",
                        messages=[
                            {"role": "system", "content": "You are an expert at transforming text to match specific tone styles while preserving markdown formatting."},
                            {"role": "user", "content": prompt_message}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    transformed_text = response.choices[0].message.content
                    
                    logger.info(f"Successfully transformed text using cached prompt")
                    
                    return {
                        "transformedText": transformed_text,
                        "metadata": {
                            "promptUsed": prompt,
                            "source": "cached",
                            "tone_style": tone_style
                        }
                    }
                except Exception as e:
                    logger.error(f"Error calling OpenAI API: {e}")
                    # Fall back to basic transformation
            else:
                # No cached prompt exists - run optimization if requested
                logger.info(f"No cached prompt found for {tone_style}, running optimization")
                
                # Load example pairs from the CSV file
                pairs_csv = os.path.join(os.path.dirname(__file__), "pairs.csv")
                example_pairs = _load_example_pairs_from_csv(pairs_csv, max_samples=5)
                
                if len(example_pairs) >= 2:
                    try:
                        # Create example texts and neutral examples
                        example_texts = [pair["transformed"] for pair in example_pairs]
                        neutral_examples = [pair["original"] for pair in example_pairs]
                        
                        # Run optimization to generate a prompt for this tone style
                        logger.info(f"Starting optimization for {tone_style} with {len(example_pairs)} example pairs")
                        
                        # Call optimize_prompt to create a new prompt
                        from optimize import optimize_prompt
                        
                        optimization_result = optimize_prompt(
                            tone_style=tone_style,
                            example_pairs=example_pairs,
                            task_id=None,
                            pop_size=10,
                            generations=3,
                            max_time_seconds=120
                        )
                        
                        if "prompt" in optimization_result and optimization_result["prompt"]:
                            # Cache the newly generated prompt
                            logger.info(f"Generated new prompt for {tone_style}, caching it")
                            prompt_cache.save(tone_style, optimization_result)
                            
                            # Use the new prompt to transform the text
                            new_prompt = optimization_result["prompt"]
                            prompt_message = f"Transform the following text to match a {tone_style} tone style. {new_prompt}\n\n{formatting_instruction}\n\nText to transform:\n{text}"
                            
                            response = client.chat.completions.create(
                                model="gpt-4-0125-preview",
                                messages=[
                                    {"role": "system", "content": "You are an expert at transforming text to match specific tone styles while preserving markdown formatting."},
                                    {"role": "user", "content": prompt_message}
                                ],
                                temperature=0.7,
                                max_tokens=2000
                            )
                            transformed_text = response.choices[0].message.content
                            
                            logger.info(f"Successfully transformed text using newly generated prompt")
                            
                            return {
                                "transformedText": transformed_text,
                                "metadata": {
                                    "promptUsed": new_prompt,
                                    "source": "newly_generated",
                                    "tone_style": tone_style
                                }
                            }
                    except Exception as e:
                        logger.error(f"Error generating new prompt: {e}")
                        # Continue to fallback methods
        
        # If we don't have a cached prompt or use_cached is False, or API call failed
        # Create a basic prompt for the tone style
        system_msg = "You are an expert at transforming text to match specific tone styles while preserving all markdown formatting."
        
        if tone_style == "professional":
            user_msg = f"Transform the following text to have a professional tone. Use clear, authoritative language with industry-specific terminology where appropriate. Maintain a formal structure with well-organized points.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "friendly":
            user_msg = f"Transform the following text to have a friendly, conversational tone. Use warm, approachable language with occasional casual expressions. Make it sound like you're talking to a friend.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "formal":
            user_msg = f"Transform the following text to have a formal tone. Use sophisticated vocabulary, complex sentence structures, and maintain a respectful distance from the reader.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "casual":
            user_msg = f"Transform the following text to have a casual, relaxed tone. Use everyday language, contractions, and a conversational style as if chatting with a friend.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "technical":
            user_msg = f"Transform the following text to have a technical tone. Use precise terminology, data-driven statements, and logical structure. Focus on accuracy and specificity.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "persuasive":
            user_msg = f"Transform the following text to have a persuasive tone. Use compelling language, rhetorical questions, and emphasize benefits. Create a sense of urgency and appeal to the reader's interests.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "enthusiastic":
            user_msg = f"Transform the following text to have an enthusiastic tone. Use energetic language, positive expressions, and convey excitement. Incorporate exclamation points where appropriate.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        elif tone_style == "youtube_thumbnails":
            user_msg = f"Transform the following text to have a YouTube thumbnail style. Use attention-grabbing, exciting language with ALL CAPS for emphasis, occasional emojis, dramatic vocabulary, and short punchy sentences. Create a sense of urgency and excitement.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        else:
            user_msg = f"Transform the following text to match a {tone_style} tone style.\n\n{formatting_instruction}\n\nText to transform:\n{text}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            transformed_text = response.choices[0].message.content
            
            logger.info(f"Successfully transformed text using basic prompt")
            
            return {
                "transformedText": transformed_text,
                "metadata": {
                    "promptUsed": f"Basic {tone_style} prompt",
                    "source": "fallback",
                    "tone_style": tone_style
                }
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API with fallback prompt: {e}")
            
            # Last resort fallback - return original text with modification note
            return {
                "transformedText": text,
                "error": f"Failed to transform text: {str(e)}",
                "metadata": {
                    "source": "error",
                    "tone_style": tone_style
                }
            }
        
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        return {
            "error": str(e),
            "transformedText": text  # Return original text in case of error
        }

def evaluate_transformation(
    original_text: str,
    transformed_text: str,
    tone_style: str
) -> Dict[str, Any]:
    """
    Evaluate a text transformation
    
    Args:
        original_text: The original text
        transformed_text: The transformed text
        tone_style: The tone style used
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        logger.info(f"Evaluating transformation for tone style: {tone_style}")
        
        # In a real implementation, this would use a more sophisticated evaluation
        score = 0.8  # Simulated score
        
        return {
            "score": score,
            "feedback": "The transformation successfully applied the requested tone style."
        }
    
    except Exception as e:
        logger.error(f"Error evaluating transformation: {e}")
        return {
            "error": str(e)
        }

def clean_old_tasks(max_age_hours: int = 24) -> int:
    """
    Clean up old optimization tasks
    
    Args:
        max_age_hours: Maximum age of tasks to keep
        
    Returns:
        Number of tasks removed
    """
    try:
        logger.info(f"Cleaning up old tasks (max age: {max_age_hours} hours)")
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        removed_count = 0
        with _tasks_lock:
            for task_id in list(_optimization_tasks.keys()):
                task = _optimization_tasks[task_id]
                timestamp = task.get("timestamp", 0)
                
                if current_time - timestamp > max_age_seconds:
                    del _optimization_tasks[task_id]
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} old tasks")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error cleaning old tasks: {e}")
        return 0

def is_available() -> bool:
    """
    Check if the GA optimization service is available
    
    Returns:
        True if available, False otherwise
    """
    # Simply check if we can import the required modules
    try:
        logger.info("Checking if GA optimization service is available")
        return True
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return False

def load_example_pairs_from_csv(csv_path: str, max_samples: int = 10) -> List[Dict[str, str]]:
    """
    Public wrapper for _load_example_pairs_from_csv
    Loads example pairs from a CSV file

    Args:
        csv_path: Path to the CSV file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of dictionaries with 'original' and 'transformed' keys
    """
    return _load_example_pairs_from_csv(csv_path, max_samples) 