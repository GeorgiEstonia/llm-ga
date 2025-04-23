"""
Contains Langchain/OpenAI logic for GA steps: Stylizer, Fitness, Crossover.
Stylizer uses Langchain ChatAnthropic.
Fitness/Crossover use OpenAI Assistants API.
"""
import os
import sys
import logging
import random
import time
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler # Import OpenAI client
from langchain_openai import ChatOpenAI # Still needed for population generation
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser # No longer used for fitness
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel as V1BaseModel, Field
from langchain_core.runnables import Runnable

# --- Logging Setup ---
logger = logging.getLogger("ga_llm_steps")
# Ensure logger is configured (might be handled in main.py, but good practice here too)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set higher level for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Environment & API Client Setup ---
load_dotenv()

# Keep LLM init functions for Population Gen (OpenAI) and Stylizer (Anthropic)
_openai_chat_llm_cache = None
_anthropic_llm_cache = None
_openai_client_cache = None # Cache for the Assistants API client

# Assistant IDs from user
FITNESS_ASSISTANT_ID = "asst_kQfwUgcVrkDZGwKBQZzgIAFG"
CROSSOVER_ASSISTANT_ID = "asst_MD8Yz7i6FcjFWXKHMyJGth5S"

# Polling constants for Assistants API
ASSISTANT_POLL_INTERVAL_S = 3
ASSISTANT_TIMEOUT_S = 300 # 5 minutes timeout for assistant run

def get_openai_chat_llm(temperature=0.7) -> ChatOpenAI:
    """Gets ChatOpenAI client (for population gen)."""
    global _openai_chat_llm_cache
    if _openai_chat_llm_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        logger.info(f"Initializing OpenAI Chat model: {model_name}")
        _openai_chat_llm_cache = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)
    # Update temperature if requested differs from cached (simple approach)
    if _openai_chat_llm_cache.temperature != temperature:
         _openai_chat_llm_cache.temperature = temperature
    return _openai_chat_llm_cache

def get_anthropic_llm(temperature=0.7) -> ChatAnthropic:
    """Gets ChatAnthropic client (for stylizer)."""
    global _anthropic_llm_cache
    if _anthropic_llm_cache is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20240620")
        logger.info(f"Initializing Anthropic model: {model_name}")
        _anthropic_llm_cache = ChatAnthropic(anthropic_api_key=api_key, model_name=model_name, temperature=temperature, max_tokens=4000)
    if _anthropic_llm_cache.temperature != temperature:
         _anthropic_llm_cache.temperature = temperature
    return _anthropic_llm_cache

def get_openai_assistant_client() -> OpenAI:
    """Gets OpenAI client configured for Assistants API."""
    global _openai_client_cache
    if _openai_client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        logger.info("Initializing OpenAI client for Assistants API...")
        _openai_client_cache = OpenAI(api_key=api_key)
    return _openai_client_cache

# --- Pydantic Model for Fitness Output --- (Still useful for defining expected structure)
class FitnessOutput(V1BaseModel):
    """Structure for the fitness evaluation output."""
    fitness_score: float = Field(..., description="The numerical fitness score from 0.0 to 10.0...")
    feedback: str = Field(..., description="Detailed qualitative feedback...")

# --- Chain Definitions & Execution Functions ---

_stylizer_chain_cache: Optional[Runnable] = None # Keep stylizer as a Langchain Runnable

def run_stylize_chain(payload: Dict[str, Any]) -> str:
    """Runs the stylizer chain (Claude via Langchain)."""
    global _stylizer_chain_cache
    logger.debug(f"Stylizer received payload keys: {list(payload.keys())}")
    try:
        if _stylizer_chain_cache is None:
            logger.info("Building Stylizer chain...")
            llm = get_anthropic_llm()
            system_prompt_template = (
                "You are an expert at replicating text styles. Given the neutral version of a text and a style prompt, "
                "your role is to transform the neutral text into the desired style. Make sure to adhere strictly to the instructions "
                "and the style of the example text provided in the prompt. Output stylized text only without any additional comments.\n\n"
                "% STYLE GUIDELINES:\n{style_prompt}\n\n"
                "% NEUTRAL TEXT EXAMPLE\n{neutralized_text_example}\n\n"
                "% STYLIZED TEXT EXAMPLE\n{stylized_text_example}"
            )
            human_prompt_template = "{neutralized_text_to_transform}"
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt_template),
                HumanMessagePromptTemplate.from_template(human_prompt_template)
            ])
            _stylizer_chain_cache = chat_prompt | llm | StrOutputParser()

        logger.debug("Invoking Stylizer chain...")
        stylized_result = _stylizer_chain_cache.invoke(payload)
        logger.info(f"Stylizer finished. Result: {stylized_result[:100]}...")
        return stylized_result or "" # Return empty string if result is None
    except Exception as e:
        logger.error(f"Error running stylizer chain: {e}", exc_info=True)
        return "" # Return empty string on error to avoid breaking GA flow

def _run_assistant(client: OpenAI, assistant_id: str, user_message: str) -> Optional[str]:
    """Helper function to run an Assistant and get the last text response."""
    try:
        thread = client.beta.threads.create()
        logger.debug(f"Created thread: {thread.id}")

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
        )
        logger.debug(f"Added user message to thread {thread.id}")

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        logger.debug(f"Created run: {run.id}")

        start_time = time.time()
        while run.status in ["queued", "in_progress"]:
            if time.time() - start_time > ASSISTANT_TIMEOUT_S:
                logger.error(f"Assistant run {run.id} timed out after {ASSISTANT_TIMEOUT_S}s.")
                client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                return None
            time.sleep(ASSISTANT_POLL_INTERVAL_S)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            logger.debug(f"Polling run {run.id}: Status = {run.status}")

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
            # Find the latest assistant message with text content
            for msg in messages.data:
                if msg.role == "assistant":
                    for content_block in msg.content:
                        if content_block.type == "text":
                            logger.debug(f"Assistant response received (Run ID: {run.id})")
                            return content_block.text.value
            logger.warning(f"Run {run.id} completed but no assistant text message found.")
            return None
        else:
            logger.error(f"Assistant run {run.id} failed or was cancelled. Status: {run.status}")
            return None

    except Exception as e:
        logger.error(f"Error interacting with Assistant API: {e}", exc_info=True)
        return None
    finally:
        # Optional: Clean up the thread if desired, otherwise it persists
        # try:
        #     if thread:
        #         client.beta.threads.delete(thread.id)
        #         logger.debug(f"Deleted thread: {thread.id}")
        # except Exception as del_e:
        #     logger.warning(f"Failed to delete thread {thread.id}: {del_e}")
        pass

def run_fitness_chain(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the fitness evaluation using the OpenAI Fitness Assistant."""
    logger.debug(f"Fitness Assistant received payload keys: {list(payload.keys())}")
    fallback_result = {"fitness_score": 0.0, "feedback": "Fitness evaluation failed (Assistant error)."}

    # Construct the user message for the assistant
    user_message = (
        f"**Neutralized Text:**\n{payload['neutral_text']}\n\n"
        f"**Benchmark Stylized Text:**\n{payload['target_stylized_text']}\n\n"
        f"**Candidate Stylized Text (to evaluate):**\n{payload['stylized_text_to_evaluate']}\n\n"
        f"Respond using JSON with 'fitness_score' (float 0.0-10.0) and 'feedback' (string) keys ONLY."
    )

    client = get_openai_assistant_client()
    assistant_response = _run_assistant(client, FITNESS_ASSISTANT_ID, user_message)

    if assistant_response:
        try:
            # Attempt to parse the JSON response from the assistant
            fitness_data = json.loads(assistant_response)
            if isinstance(fitness_data, dict) and "fitness_score" in fitness_data and "feedback" in fitness_data:
                # Basic validation
                score = max(0.0, min(10.0, float(fitness_data["fitness_score"])))
                feedback = str(fitness_data["feedback"])
                logger.info(f"Fitness evaluation finished. Score: {score}")
                return {"fitness_score": score, "feedback": feedback}
            else:
                logger.error(f"Assistant {FITNESS_ASSISTANT_ID} response was not the expected JSON dict. Got: {assistant_response}")
                return {**fallback_result, "feedback": "Fitness evaluation failed (Invalid JSON format from Assistant)."}
        except json.JSONDecodeError:
            logger.error(f"Assistant {FITNESS_ASSISTANT_ID} response was not valid JSON. Got: {assistant_response}")
            return {**fallback_result, "feedback": "Fitness evaluation failed (JSON decode error from Assistant)."}
        except Exception as e:
             logger.error(f"Error processing fitness assistant response: {e}", exc_info=True)
             return fallback_result
    else:
        logger.error(f"Assistant {FITNESS_ASSISTANT_ID} failed to provide a response.")
        return fallback_result

def run_crossover_chain(payload: Dict[str, Any]) -> str:
    """Runs the crossover using the OpenAI Crossover Assistant."""
    logger.debug(f"Crossover Assistant received payload keys: {list(payload.keys())}")

    # Construct the user message for the assistant
    user_message = (
        f"**Neutral Text Example (for context):**\n{payload['neutralized_text_example']}\n\n"
        f"**Stylized Text Example (for context):**\n{payload['stylized_text_example']}\n\n"
        f"**Parent Prompt 1:**\n{payload['prompt_1']}\n"
        f"**Fitness Score 1:** {payload['prompt_1_fitness_score']}\n"
        f"**Feedback 1:** {payload['prompt_1_fitness_feedback']}\n\n"
        f"**Parent Prompt 2:**\n{payload['prompt_2']}\n"
        f"**Fitness Score 2:** {payload['prompt_2_fitness_score']}\n"
        f"**Feedback 2:** {payload['prompt_2_fitness_feedback']}\n\n"
        f"Output *only* the new child instruction prompt text, with no explanations or commentary."
    )

    client = get_openai_assistant_client()
    child_prompt = _run_assistant(client, CROSSOVER_ASSISTANT_ID, user_message)

    if child_prompt:
        logger.info(f"Crossover finished. Resulting prompt: {child_prompt[:100]}...")
        return child_prompt.strip()
    else:
        logger.error(f"Assistant {CROSSOVER_ASSISTANT_ID} failed to provide a response.")
        return "" # Return empty string on error 