import logging
import json
import random
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
import os

# Setup logging
logger = logging.getLogger("llm_ga")
logger.setLevel(logging.DEBUG)

# Add file handler if not already added
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    file_handler = logging.FileHandler("ga_optimization.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Function to generate new prompts for the population
def generate_prompts(
    tone_style: str,
    example_pairs: List[Dict[str, str]],
    population_size: int = 10
) -> List[str]:
    """
    Generate an initial population of prompts for the genetic algorithm
    
    Args:
        tone_style: The tone style to optimize for
        example_pairs: List of dictionaries with 'original' and 'transformed' text pairs
        population_size: Size of the population to generate
        
    Returns:
        List of prompt strings
    """
    logger.info(f"Generating initial population of {population_size} prompts for tone: {tone_style}")
    logger.info(f"Using {len(example_pairs)} example pairs")
    
    prompts = []
    
    # Fixed system message components to choose from
    system_messages = [
        f"You are an expert at writing in a {tone_style} tone.",
        f"You specialize in transforming text to sound {tone_style}.",
        f"You excel at rewriting content to match a {tone_style} style.",
        f"Your expertise is in adapting text to have a {tone_style} tone while preserving meaning."
    ]
    
    # Fixed instruction components to choose from
    instructions = [
        f"Transform the following text to match a {tone_style} tone.",
        f"Rewrite the text below to sound more {tone_style}.",
        f"Convert this content to have a {tone_style} style.",
        f"Adapt the following text to have a {tone_style} tone while preserving the original meaning."
    ]
    
    # Additional components that can be included
    additional_instructions = [
        "Keep the same information and meaning, but change the tone.",
        "Preserve all important details while adjusting the tone.",
        "Focus on maintaining the core message while changing the style.",
        "Ensure the transformed text contains all the original information.",
        f"Make it sound {tone_style} without losing any key points.",
        f"Emphasize the {tone_style} aspects while retaining all facts."
    ]
    
    # Generate prompts using different combinations
    for _ in range(population_size):
        system = random.choice(system_messages)
        instruction = random.choice(instructions)
        additional = random.sample(additional_instructions, k=random.randint(1, 3))
        
        # Decide whether to include examples
        include_examples = random.choice([True, False])
        
        # Build the prompt
        prompt_parts = [system, instruction]
        prompt_parts.extend(additional)
        
        if include_examples and example_pairs:
            prompt_parts.append("\nHere are some examples:")
            
            # Choose a random number of examples to include
            num_examples = min(random.randint(1, 3), len(example_pairs))
            examples = random.sample(example_pairs, k=num_examples)
            
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Original: {example['original']}")
                prompt_parts.append(f"Transformed: {example['transformed']}")
                
        prompt_parts.append("\nNow transform this text:")
        prompt_parts.append("{text}")
        
        prompt = "\n\n".join(prompt_parts)
        prompts.append(prompt)
        
        logger.debug(f"Generated prompt #{len(prompts)}, length: {len(prompt)} chars")
    
    logger.info(f"Successfully generated {len(prompts)} prompts")
    return prompts

# Evaluate a single prompt
def evaluate_prompt(
    prompt: str,
    example_pairs: List[Dict[str, str]],
    evaluation_pairs: List[Dict[str, str]]
) -> float:
    """
    Evaluate a prompt using a fitness function
    
    Args:
        prompt: The prompt to evaluate
        example_pairs: The example pairs used for training
        evaluation_pairs: The pairs used for evaluation (not seen during training)
        
    Returns:
        A fitness score (higher is better)
    """
    try:
        logger.info(f"Evaluating prompt (length: {len(prompt)} chars)")
        
        # Check if we should use real evaluation with OpenAI API
        use_real_eval = os.environ.get("USE_REAL_EVAL", "").lower() == "true"
        
        if use_real_eval:
            try:
                import openai
                from openai import OpenAI
                
                client = OpenAI()
                logger.info("Using real OpenAI API evaluation")
                
                # Evaluate with actual API calls - use just 2 examples to save tokens
                test_pairs = evaluation_pairs[:2] if len(evaluation_pairs) > 2 else evaluation_pairs
                scores = []
                
                for pair in test_pairs:
                    try:
                        # Format the prompt with the test example
                        formatted_prompt = prompt.replace("{text}", pair["original"])
                        
                        # Make the API call
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",  # Use a less expensive model for evaluations
                            messages=[
                                {"role": "system", "content": "You are an assistant that helps evaluate text transformations."},
                                {"role": "user", "content": formatted_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        
                        # Get the transformed text from the response
                        transformed_text = response.choices[0].message.content
                        
                        # Now evaluate the similarity to the target (stylized) text
                        # We'll use another API call to compare them
                        eval_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an assistant that evaluates how well a text matches a target style."},
                                {"role": "user", "content": f"""On a scale of 0.0 to 1.0, how well does the following text match the style of the target text? 
                                
                                Original text: {pair["original"]}
                                
                                Target (stylized) text: {pair["transformed"]}
                                
                                Generated text: {transformed_text}
                                
                                Score the match in terms of style, tone, and flow ONLY (not content accuracy). 
                                Return ONLY a number between 0.0 and 1.0 as your answer, with no other text."""}
                            ],
                            temperature=0.1,
                            max_tokens=10
                        )
                        
                        # Extract the score from the response
                        score_text = eval_response.choices[0].message.content.strip()
                        try:
                            score = float(score_text)
                            # Ensure score is between 0 and 1
                            score = max(0.0, min(1.0, score))
                        except ValueError:
                            logger.warning(f"Could not parse score from response: {score_text}")
                            score = 0.5  # Default score if parsing fails
                        
                        scores.append(score)
                        logger.info(f"Evaluated example with score: {score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating example: {e}")
                        scores.append(0.3)  # Penalty score for failed evaluation
                
                # Calculate average score
                avg_score = sum(scores) / len(scores) if scores else 0
                logger.info(f"Real API evaluation complete. Average fitness score: {avg_score:.4f}")
                return avg_score
                
            except ImportError:
                logger.warning("OpenAI library not available, falling back to simulated evaluation")
                use_real_eval = False
            except Exception as e:
                logger.error(f"Error in real evaluation: {e}")
                use_real_eval = False
        
        # Simulated evaluation (fallback or when real evaluation is disabled)
        if not use_real_eval:
            logger.info("Using simulated evaluation")
            scores = []
            
            for pair in evaluation_pairs:
                # Check if the prompt has certain desirable features
                has_clear_instruction = any(x in prompt.lower() for x in ["transform", "rewrite", "convert"])
                has_preserve_meaning = "preserv" in prompt.lower() or "maintaining" in prompt.lower()
                has_examples = "example" in prompt.lower()
                
                # Base score is random
                score = random.random() * 0.5  # Base random component
                
                # Add bonuses for good features
                if has_clear_instruction:
                    score += 0.1
                if has_preserve_meaning:
                    score += 0.15
                if has_examples:
                    score += 0.1
                
                # Length penalty (too short or too long)
                if len(prompt) < 100:
                    score -= 0.1
                elif len(prompt) > 1000:
                    score -= 0.1
                
                # Add some randomness
                score += random.random() * 0.15
                
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
                scores.append(score)
            
            # Calculate average score across all evaluation pairs
            avg_score = sum(scores) / len(scores) if scores else 0
            logger.info(f"Simulated evaluation complete. Fitness score: {avg_score:.4f}")
            return avg_score
        
    except Exception as e:
        logger.error(f"Error evaluating prompt: {e}", exc_info=True)
        return 0.0  # Assign minimum fitness for failed evaluations

# Select parents for the next generation
def select_parents(
    population: List[str],
    fitness_scores: List[float],
    num_parents: int
) -> List[str]:
    """
    Select parents for the next generation using tournament selection
    
    Args:
        population: The current population of prompts
        fitness_scores: The corresponding fitness scores
        num_parents: Number of parents to select
        
    Returns:
        List of selected parent prompts
    """
    logger.info(f"Selecting {num_parents} parents from population of {len(population)}")
    
    # Create pairs of (prompt, fitness score)
    population_with_fitness = list(zip(population, fitness_scores))
    
    # Sort by fitness score (descending)
    population_with_fitness.sort(key=lambda x: x[1], reverse=True)
    
    # Select the best individuals
    selected_parents = [p[0] for p in population_with_fitness[:num_parents]]
    
    logger.info(f"Selected {len(selected_parents)} parents. Top fitness: {population_with_fitness[0][1]:.4f}")
    return selected_parents

# Create new individuals through crossover
def crossover(
    parents: List[str],
    crossover_rate: float = 0.8,
    population_size: int = 10
) -> List[str]:
    """
    Create a new generation through crossover
    
    Args:
        parents: The selected parent prompts
        crossover_rate: Probability of crossover
        population_size: Target size for the new population
        
    Returns:
        New population after crossover
    """
    logger.info(f"Performing crossover with {len(parents)} parents to create {population_size} offspring")
    
    # Always keep the best parents (elitism)
    new_population = parents[:2].copy()
    
    # Generate the rest through crossover
    while len(new_population) < population_size:
        # Select two random parents
        parent1, parent2 = random.sample(parents, 2)
        
        if random.random() < crossover_rate:
            # Split each parent into sentences or paragraphs
            parent1_parts = parent1.split("\n\n")
            parent2_parts = parent2.split("\n\n")
            
            # Choose a random crossover point for each parent
            crossover_point1 = random.randint(1, max(1, len(parent1_parts) - 1))
            crossover_point2 = random.randint(1, max(1, len(parent2_parts) - 1))
            
            # Create two new offspring
            offspring1 = "\n\n".join(parent1_parts[:crossover_point1] + parent2_parts[crossover_point2:])
            offspring2 = "\n\n".join(parent2_parts[:crossover_point2] + parent1_parts[crossover_point1:])
            
            # Ensure "{text}" placeholder is present
            if "{text}" not in offspring1:
                offspring1 += "\n\n{text}"
            if "{text}" not in offspring2:
                offspring2 += "\n\n{text}"
            
            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)
        else:
            # If no crossover, just add the parents
            new_population.append(parent1)
            if len(new_population) < population_size:
                new_population.append(parent2)
    
    logger.info(f"Crossover complete. New population size: {len(new_population)}")
    return new_population[:population_size]

# Mutate the population
def mutate(
    population: List[str],
    mutation_rate: float = 0.2,
    tone_style: str = ""
) -> List[str]:
    """
    Apply mutation to the population
    
    Args:
        population: The population to mutate
        mutation_rate: Probability of mutation
        tone_style: The tone style for context
        
    Returns:
        Mutated population
    """
    logger.info(f"Applying mutations with rate {mutation_rate} to population of {len(population)}")
    
    # Mutation operations
    mutation_ops = [
        "add_instruction",
        "remove_instruction",
        "replace_instruction",
        "change_format"
    ]
    
    # Additional instructions that can be added
    additional_instructions = [
        f"Make sure the text sounds {tone_style} but natural.",
        f"Maintain the {tone_style} voice throughout.",
        "Preserve all important facts and details.",
        "Don't add any new information not found in the original.",
        "Ensure the core message remains clear.",
        f"Adapt the vocabulary to better match a {tone_style} tone.",
        "Keep sentence structures that work well, but change those that don't fit the tone."
    ]
    
    mutated_population = []
    
    for prompt in population:
        if random.random() < mutation_rate:
            logger.debug(f"Mutating prompt (original length: {len(prompt)})")
            
            # Choose a random mutation operation
            op = random.choice(mutation_ops)
            
            if op == "add_instruction":
                # Add a random instruction
                new_instruction = random.choice(additional_instructions)
                parts = prompt.split("\n\n")
                insert_position = random.randint(1, max(1, len(parts) - 1))
                parts.insert(insert_position, new_instruction)
                prompt = "\n\n".join(parts)
                
            elif op == "remove_instruction" and prompt.count("\n\n") > 2:
                # Remove a random instruction (but keep the {text} part)
                parts = prompt.split("\n\n")
                text_part_index = None
                for i, part in enumerate(parts):
                    if "{text}" in part:
                        text_part_index = i
                        break
                
                # Only remove if there are parts that don't contain {text}
                removable_parts = [i for i in range(len(parts)) if i != text_part_index]
                if removable_parts:
                    remove_index = random.choice(removable_parts)
                    parts.pop(remove_index)
                    prompt = "\n\n".join(parts)
                
            elif op == "replace_instruction":
                # Replace a random instruction
                parts = prompt.split("\n\n")
                text_part_index = None
                for i, part in enumerate(parts):
                    if "{text}" in part:
                        text_part_index = i
                        break
                
                # Only replace if there are parts that don't contain {text}
                replaceable_parts = [i for i in range(len(parts)) if i != text_part_index]
                if replaceable_parts:
                    replace_index = random.choice(replaceable_parts)
                    parts[replace_index] = random.choice(additional_instructions)
                    prompt = "\n\n".join(parts)
                
            elif op == "change_format":
                # Change the format
                parts = prompt.split("\n\n")
                random.shuffle(parts)
                
                # Ensure {text} appears at the end
                text_part = None
                for i, part in enumerate(parts):
                    if "{text}" in part:
                        text_part = parts.pop(i)
                        break
                
                if text_part:
                    parts.append(text_part)
                else:
                    parts.append("{text}")
                
                prompt = "\n\n".join(parts)
            
            logger.debug(f"Mutation '{op}' applied. New length: {len(prompt)}")
        
        # Ensure prompt has the {text} placeholder
        if "{text}" not in prompt:
            prompt += "\n\n{text}"
            
        mutated_population.append(prompt)
    
    logger.info(f"Mutation complete. Mutated population size: {len(mutated_population)}")
    return mutated_population

# Run the genetic algorithm
def run_genetic_algorithm(
    tone_style: str,
    example_pairs: List[Dict[str, str]],
    generations: int = 5,
    population_size: int = 10,
    max_time_seconds: int = 600
) -> Dict[str, Any]:
    """
    Run the genetic algorithm to find an optimal prompt
    
    Args:
        tone_style: The tone style to optimize for
        example_pairs: List of dictionaries with 'original' and 'transformed' text pairs
        generations: Number of generations to run
        population_size: Size of the population
        max_time_seconds: Maximum time to run in seconds
        
    Returns:
        Dictionary with the best prompt and its fitness score
    """
    logger.info(f"Starting genetic algorithm for tone style: {tone_style}")
    logger.info(f"Parameters: generations={generations}, population_size={population_size}, max_time_seconds={max_time_seconds}")
    logger.info(f"Example pairs: {len(example_pairs)}")
    
    start_time = time.time()
    
    # Validate inputs
    if not example_pairs or len(example_pairs) < 2:
        logger.error(f"Not enough example pairs provided. Need at least 2, got {len(example_pairs)}")
        return {
            "error": "Not enough example pairs provided",
            "prompt": None,
            "fitness_score": 0
        }
    
    # Split the examples into training and evaluation sets
    random.shuffle(example_pairs)
    split_point = max(1, len(example_pairs) // 2)
    training_pairs = example_pairs[:split_point]
    evaluation_pairs = example_pairs[split_point:]
    
    logger.info(f"Split example pairs: {len(training_pairs)} for training, {len(evaluation_pairs)} for evaluation")
    
    # Generate initial population
    population = generate_prompts(tone_style, training_pairs, population_size)
    
    # Track the best prompt
    best_prompt = None
    best_fitness = 0
    generations_without_improvement = 0
    
    # Run the algorithm
    for generation in range(generations):
        if time.time() - start_time > max_time_seconds:
            logger.warning(f"Time limit reached after {generation} generations")
            break
            
        logger.info(f"Starting generation {generation+1}/{generations}")
        
        # Evaluate the population
        fitness_scores = []
        for i, prompt in enumerate(population):
            fitness = evaluate_prompt(prompt, training_pairs, evaluation_pairs)
            fitness_scores.append(fitness)
            logger.debug(f"Prompt #{i+1} fitness: {fitness:.4f}")
        
        # Find the best prompt in this generation
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        current_best_prompt = population[max_fitness_idx]
        current_best_fitness = fitness_scores[max_fitness_idx]
        
        logger.info(f"Generation {generation+1} complete")
        logger.info(f"Best fitness in generation: {current_best_fitness:.4f}")
        logger.info(f"Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
        
        # Update the overall best if needed
        if current_best_fitness > best_fitness:
            best_prompt = current_best_prompt
            best_fitness = current_best_fitness
            generations_without_improvement = 0
            logger.info(f"New best prompt found! Fitness: {best_fitness:.4f}")
        else:
            generations_without_improvement += 1
            logger.info(f"No improvement for {generations_without_improvement} generations")
        
        # Early stopping if no improvement for 3 generations
        if generations_without_improvement >= 3:
            logger.info("Early stopping: No improvement for 3 generations")
            break
            
        # Create the next generation
        parents = select_parents(population, fitness_scores, num_parents=max(3, population_size // 3))
        population = crossover(parents, population_size=population_size)
        population = mutate(population, tone_style=tone_style)
    
    total_time = time.time() - start_time
    logger.info(f"Genetic algorithm completed in {total_time:.2f} seconds")
    logger.info(f"Best prompt fitness: {best_fitness:.4f}")
    
    # Return the best prompt and its fitness score
    return {
        "prompt": best_prompt,
        "fitness_score": best_fitness
    }

# Main optimization function
def optimize_prompt(
    tone_style: str,
    example_pairs: List[Dict[str, str]],
    task_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize a prompt for a specific tone style
    
    Args:
        tone_style: The tone style to optimize for
        example_pairs: List of dictionaries with 'original' and 'transformed' text pairs
        task_id: Optional task ID for tracking
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with the optimized prompt and metadata
    """
    if task_id is None:
        task_id = str(uuid.uuid4())
        
    logger.info(f"Starting prompt optimization for tone style: {tone_style}, task_id: {task_id}")
    
    try:
        # Set algorithm parameters
        generations = kwargs.get("generations", 5)
        population_size = kwargs.get("population_size", 10)
        max_time_seconds = kwargs.get("max_time_seconds", 600)
        
        # Run the genetic algorithm
        result = run_genetic_algorithm(
            tone_style=tone_style,
            example_pairs=example_pairs,
            generations=generations,
            population_size=population_size,
            max_time_seconds=max_time_seconds
        )
        
        # Log the result
        if "error" in result:
            logger.error(f"Optimization failed: {result['error']}")
            return {
                "status": "error",
                "error": result["error"],
                "task_id": task_id
            }
            
        logger.info(f"Optimization succeeded. Fitness score: {result['fitness_score']:.4f}")
        logger.info(f"Prompt length: {len(result['prompt'])} chars")
        
        # Add metadata
        result["tone_style"] = tone_style
        result["task_id"] = task_id
        result["status"] = "success"
        result["timestamp"] = time.time()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in optimize_prompt: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "task_id": task_id
        } 