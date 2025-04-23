"""
Generates the initial prompt population using Langchain and OpenAI.
Runs 7 different prompt creation workflows in parallel.
"""
import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableParallel, RunnableLambda

# Configure logging (same format as main.py for consistency)
logging.basicConfig(
    level=logging.INFO, # Default to INFO, can be overridden
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Set higher level for noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# --- Specific Prompt Creator Chains ---

def get_creator_1_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 1: Category-Specific Comparison"""
    logging.debug("Creating Creator 1 Chain")
    # Step 1: Generate characteristics list
    prompt1 = ChatPromptTemplate.from_template(
        "Generate a thorough list of text style characteristics (e.g., tone, register, sentence structure, vocabulary, etc.) "
        "that can be used to compare two authors who share the same category, tone, and target audience. "
        "The list must be category-specific. Include precise definitions for each characteristic to capture subtle differences "
        "in their writing approaches. Provide only the list of characteristics and their descriptions, with no additional commentary.\n\n"
        "Category:\n{category}"
    )
    chain1 = prompt1 | llm | StrOutputParser()

    # Step 2: Evaluate stylized text using the list
    prompt2 = ChatPromptTemplate.from_template(
        "Use the following list of text style characteristics to evaluate the provided text by a single author. "
        "For each characteristic, give a detailed analysis of how it manifests in the text, including specific examples "
        "that highlight unique nuances. Output only the list of characteristics and their evaluations, with no additional commentary, "
        "ensuring enough granularity to distinguish between authors within the same category who share a similar style "
        "only relying on your evaluation.\n\n"
        "Category:\n{category}\n\n"
        "List of characteristics:\n{characteristics_list}\n\n"
        "Text to evaluate:\n{stylized_text}"
    )
    chain2 = prompt2 | llm | StrOutputParser()

    # Combine steps
    def run_sequence(input_dict: Dict[str, Any]) -> str:
        char_list = chain1.invoke({"category": input_dict["category"]})
        logging.debug(f"Creator 1 - Characteristics List: {char_list[:100]}...")
        final_input = {
            "category": input_dict["category"],
            "characteristics_list": char_list,
            "stylized_text": input_dict["stylized_text"]
        }
        return chain2.invoke(final_input)

    return RunnableLambda(run_sequence)

def get_creator_2_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 2: Normalized vs Stylized Text Comparison"""
    logging.debug("Creating Creator 2 Chain")
    # Step 1: Generate characteristics list
    prompt1 = ChatPromptTemplate.from_template(
        "Given examples of a normalized text and its stylized counterpart, do not evaluate those specific samples. "
        "Instead, generate a comprehensive list of text style characteristics (e.g., tone, register, sentence structure, "
        "vocabulary, etc. but specific to those texts) that can be used to compare and assess differences between normalized "
        "and stylized texts similar to the examples. Clearly define each characteristic and ensure they are sufficient to guide "
        "a transformation from a normalized text similar to the example text provided into a stylized version similar to the "
        "example provided. Provide only the list of characteristics and their definitions, with no additional commentary.\n\n"
        "Normalized text:\n{normalized_text}\n\n"
        "Stylized text:\n{stylized_text}"
    )
    chain1 = prompt1 | llm | StrOutputParser()

    # Step 2: Compare texts using the list
    prompt2 = ChatPromptTemplate.from_template(
        "Use the following list of text style characteristics to compare a normalized text with its stylized counterpart. "
        "For each characteristic, provide a detailed analysis of how the stylized text differs from the normalized text, "
        "including specific examples that highlight unique nuances. Output only the list of characteristics and their evaluations, "
        "with no additional commentary. Ensure enough granularity to distinguish differences in style even when both texts fall "
        "under the same category and share a similar tone.\n\n"
        "List of characteristics:\n{characteristics_list}\n\n"
        "Normalized text:\n{normalized_text}\n\n"
        "Stylized text:\n{stylized_text}"
    )
    chain2 = prompt2 | llm | StrOutputParser()

    # Combine steps
    def run_sequence(input_dict: Dict[str, Any]) -> str:
        char_list = chain1.invoke({
            "normalized_text": input_dict["normalized_text"],
            "stylized_text": input_dict["stylized_text"]
        })
        logging.debug(f"Creator 2 - Characteristics List: {char_list[:100]}...")
        final_input = {
            "characteristics_list": char_list,
            "normalized_text": input_dict["normalized_text"],
            "stylized_text": input_dict["stylized_text"]
        }
        return chain2.invoke(final_input)

    return RunnableLambda(run_sequence)

def get_creator_3_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 3: Defining Characteristics of the Stylized Text"""
    logging.debug("Creating Creator 3 Chain")
    prompt = ChatPromptTemplate.from_template(
        "Analyze the provided normalized text and its stylized counterpart, then create a list of text style characteristics "
        "that highlight the unique stylistic elements introduced in the stylized version. For each characteristic, define it briefly "
        "and provide a comparative evaluation that illustrates how it differs from the normalized text. Output only the final list "
        "of characteristics and their evaluations, with no additional commentary, ensuring sufficient detail to distinguish "
        "the style changes introduced in the stylized text.\n\n"
        "Normalized text:\n{normalized_text}\n\n"
        "Stylized text:\n{stylized_text}"
    )
    return prompt | llm | StrOutputParser()

def get_creator_4_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 4: Evaluation Against Default Characteristics"""
    logging.debug("Creating Creator 4 Chain")
    # Note: The list is long, embedding directly. Consider loading from a file if preferred.
    default_characteristics = """
- **Vocabulary Choice**: Specific word preferences, use of jargon, formality level, and any idiosyncratic words or phrases.
- **Sentence Structure**: Preference for short or long sentences, level of complexity (e.g., nested clauses), and punctuation patterns.
- **Tone and Register**: Formal vs. informal language, level of politeness, directness, or emotive intensity.
- **Figurative Language**: Use of metaphors, similes, hyperbole, or idiomatic expressions.
- **Rhetorical Devices**: Repetition, parallelism, rhetorical questions, or unique stylistic flourishes.
- **Lexical Density and Variation**: Range of distinct words used, frequency of certain word categories (e.g., adjectives, adverbs), and the overall complexity of vocabulary.
- **Pacing and Rhythm**: Cadence created by punctuation, paragraph length, and sentence flow.
- **Coherence and Cohesion**: Transitions between sentences, use of linking words, and how ideas are logically connected.
- **Pronoun and Voice Usage**: Active vs. passive voice, frequency of first-person or second-person pronouns, consistent point of view.
- **Emphasis and Intensifiers**: Use of intensifiers (e.g., "very," "absolutely"), exclamation marks, or capitalization for stress.
- **Lexical Bundles/Collocations**: Common phrases or word combinations the author repeatedly uses.
- **Overall Structure and Organization**: Paragraphing style, inclusion of headings or bullet points, and approach to introductions/conclusions.
- **Spelling and Typographical Habits**: Consistent use of American vs. British spelling, hyphenation choices, capitalization practices, or frequent typos unique to an author.
- **Synonym Selection**: Tendency to choose specific synonyms over more common alternatives, or reliance on a limited set of words.
- **Preferred Punctuation**: Overuse or avoidance of commas, semicolons, parentheses, dashes, or ellipses.
- **Use of Parenthetical Remarks and Asides**: Frequency and style of inserting personal commentary or clarifications in parentheses or between dashes.
- **Hedging and Qualifiers**: Phrases like "might be," "seems like," or "in some cases," indicating caution or uncertainty.
- **Direct vs. Indirect Speech**: Whether the author often quotes verbatim or prefers paraphrased summaries.
- **Use of Humor, Irony, or Sarcasm**: Level of wit, satire, or ironic statements in the writing.
- **Cultural and Topical References**: Pop culture nods, historical or literary references, brand mentions, or domain-specific examples.
- **Consistency of Voice**: Tendency to remain in one narrative perspective or switch (e.g., mixing first-person and third-person).
- **Stylistic Repetition or Catchphrases**: Signature slogans, repeated motifs, or inside references that appear throughout the text.
- **Register and Formality Fluctuations**: Shifts between casual language and sudden formal or academic expressions.
- **Clarity and Conciseness**: Tendency to be verbose or succinct, level of redundancy in explanations.
- **Argumentative Structure**: Whether the text includes a clear thesis, supporting points, counterarguments, or concluding remarks.
- **Use of Transitional Words and Phrases**: Patterns in connecting ideas (e.g., "however," "furthermore," "likewise").
- **Em-dash and Ellipsis Usage**: Preferred ways of creating suspense or breaks in thought.
- **Overarching Narrative or Flow**: Whether the text follows a logical progression or tends toward digressions and tangents.
- **Emotive Language**: Degree of emotional expression, adjectives or adverbs conveying sentiment, or exclamation marks.
- **Use of Questions and Exclamations**: Frequency of rhetorical or literal questions, exclamatory remarks.
- **Reference Style**: Citation formats, namedropping of authors or experts, in-text references, or footnotes.
- **Signature Openings or Closings**: Recurrent phrases that initiate or conclude sections or the entire piece.
"""
    prompt = ChatPromptTemplate.from_template(
        "Use the following list of text style characteristics to evaluate the provided text. For each characteristic, give a detailed "
        "analysis of how it manifests in the text, including specific examples that highlight unique nuances. Output only the list "
        "of characteristics and their evaluations, with no additional commentary, ensuring enough granularity to distinguish between "
        "authors within the same category who share a similar style only relying on your evaluation.\n\n"
        "List of characteristics:\n{default_characteristics}\n\n"
        "Text to evaluate:\n{stylized_text}"
    )
    # Pre-fill the default characteristics
    bound_prompt = prompt.partial(default_characteristics=default_characteristics)
    return bound_prompt | llm | StrOutputParser()

def get_creator_5_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 5: Famous Figure Resemblance"""
    logging.debug("Creating Creator 5 Chain")
    prompt = ChatPromptTemplate.from_template(
        "Your goal is to identify which authors, public figures, organizations, or writers sound most similar to the text below. "
        "Output a list of 1-3 figures or organizations that have the most similar tone, style of speech or text to the example below. "
        "Do not output anything else besides the names.\n\n"
        "Text to evaluate:\n{stylized_text}"
    )
    return prompt | llm | StrOutputParser()

def get_creator_6_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 6: General Style Directions"""
    logging.debug("Creating Creator 6 Chain")
    prompt = ChatPromptTemplate.from_template(
        "Using the example of a normalized text and its stylized counterpart, derive a comprehensive set of transformation instructions "
        "that could be applied to any similar normalized text to achieve a comparable stylized effect. Focus on identifying the style shifts "
        "(e.g., tone, vocabulary, syntax, and other relevant elements) needed to replicate the stylized outcome. Provide only these "
        "step-by-step style transformation instructions with no additional commentary.\n\n"
        "Normalized text: {normalized_text}\n\n"
        "Stylized text: {stylized_text}"
    )
    return prompt | llm | StrOutputParser()

def get_creator_7_chain(llm: ChatOpenAI) -> Runnable:
    """Creator 7: Specific Word Transformations"""
    logging.debug("Creating Creator 7 Chain")
    prompt = ChatPromptTemplate.from_template(
        "Compare the normalized text to its stylized version and identify any specific words or phrases that were altered, replaced, or removed. "
        "For each identified transformation, present it in the format <normalized_phrase> -> <stylized_phrase>. Output only this list of "
        "transformations, with no additional commentary.\n\n"
        "Normalized text: {normalized_text}\n\n"
        "Stylized text: {stylized_text}"
    )
    return prompt | llm | StrOutputParser()


# --- Main Generation Function ---

def generate_initial_population(pop_size: int, text_category: str, neutral_example: str, stylized_example: str) -> List[str]:
    """
    Generates an initial population of 7 prompts by running 7 distinct creator workflows in parallel.

    Args:
        pop_size: The desired population size (Note: this function will always attempt to generate 7 prompts).
        text_category: The category of text the prompts are for.
        neutral_example: An example of neutral text.
        stylized_example: An example of the target stylized text.

    Returns:
        A list of generated prompt strings (up to 7).
    """
    if pop_size != 7:
        logging.warning(f"Requested pop_size={pop_size}, but this function is hardcoded to generate 7 prompts via parallel workflows.")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # Default to gpt-4o
    logging.info(f"Using OpenAI model: {model_name}")
    # Adjust temperature as needed for creativity vs consistency
    llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.7)

    # Input data for all chains
    input_data = {
        "category": text_category,
        "normalized_text": neutral_example,
        "stylized_text": stylized_example,
    }

    logging.info("Building parallel prompt creator chains...")
    # Create all chains
    creator_chains = {
        "creator_1": get_creator_1_chain(llm),
        "creator_2": get_creator_2_chain(llm),
        "creator_3": get_creator_3_chain(llm),
        "creator_4": get_creator_4_chain(llm),
        "creator_5": get_creator_5_chain(llm),
        "creator_6": get_creator_6_chain(llm),
        "creator_7": get_creator_7_chain(llm),
    }

    # Create the parallel execution runnable
    parallel_runnable = RunnableParallel(creator_chains)

    logging.info("Invoking parallel chains to generate initial population...")
    generated_prompts: List[str] = []
    try:
        # Invoke all chains in parallel
        parallel_results = parallel_runnable.invoke(input_data)

        # Process results
        logging.info("Parallel execution finished. Processing results...")
        for name, result in parallel_results.items():
            if result and isinstance(result, str) and result.strip():
                cleaned_prompt = result.strip()
                generated_prompts.append(cleaned_prompt)
                logging.debug(f"Successfully processed result from {name}: {cleaned_prompt[:100]}...")
            else:
                logging.warning(f"{name} returned empty or invalid output: {repr(result)}")

    except Exception as e:
        logging.error(f"Error during parallel chain execution: {e}", exc_info=True)
        # Depending on requirements, you might want to return partial results or raise

    logging.info(f"Generated {len(generated_prompts)} prompts out of 7 workflows.")
    return generated_prompts

# Example usage (for testing this file directly)
if __name__ == '__main__':
    # Set logging level to DEBUG for detailed chain info during testing
    logging.getLogger().setLevel(logging.DEBUG) 

    print("Testing generate_initial_population...")
    # Make sure you have a .env file with OPENAI_API_KEY set
    test_prompts = generate_initial_population(
        pop_size=7, # Ignored, will generate 7
        text_category="Marketing Email Subject Lines",
        neutral_example="Newsletter Update - Week 3",
        stylized_example="✨ Your Weekly Spark: Don't Miss What's Inside! ✨"
    )
    print("\n--- Generated Prompts ---")
    if test_prompts:
        for i, p in enumerate(test_prompts):
            print(f"{i+1}: {p}")
    else:
        print("No prompts were generated successfully.")
    print("-------------------------") 