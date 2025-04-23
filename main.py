# GA‑style LLM Prompt Optimizer (CSV loader, live‑CSV, verbose)
# -----------------------------------------------------------------
# • robust Make.com async handling (polls 202 Location / Execution‑URL)
# • read‑timeout now long enough for heavy scenarios
# • prevents roulette lock‑up, clones elite if crossover under‑produces
# • optional cull by min_fitness OR survival_ratio
# • writes / updates CSV after every generation
# -----------------------------------------------------------------

from __future__ import annotations
import csv, json, logging, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

# Import the generator function
from generate_population import generate_initial_population
# Import the LLM step functions
from ga_llm_steps import run_stylize_chain, run_fitness_chain, run_crossover_chain

# ── endpoints ──
# No longer needed, using direct calls

# ── time‑outs & retries ──
# These might not be directly applicable to Langchain calls in the same way,
# but we keep BACKOFF_S for potential future retry logic within the LLM steps
# REQ_RETRIES   = 3
# CONNECT_TO_S  = 10
# READ_TO_S     = 600
BACKOFF_S     = 1.5
# ASYNC_POLL_S  = 600

CSV_PATH_OUT  = Path("prompt_evolution.csv")
PAIRS_CSV_IN  = Path("pairs.csv")

# ── logging ──
# Configure logging (main script level)
logging.basicConfig(
    level=logging.INFO, # Default to INFO
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main_ga")
# Control verbosity of sub-modules
logging.getLogger("generate_population").setLevel(logging.INFO)
logging.getLogger("ga_llm_steps").setLevel(logging.INFO)

# Remove requests session
# session = requests.Session()
# session.headers.update({"Content-Type": "application/json"})


def _j(obj: Any, max_len: int = 600) -> str:
    # Simplified: just used for logging payloads before, not strictly needed now
    # but can keep for potential future debug logging if needed
    try:
        txt = json.dumps(obj, ensure_ascii=False)
        return txt if len(txt) <= max_len else txt[:max_len] + "...(truncated)"
    except TypeError:
        return str(obj)[:max_len] # Fallback for non-serializable objects

# ─────────────────────────── file/CSV helpers ────────────────────────────
def load_pairs(path: Path) -> List[Tuple[str, str]]:
    logger.info(f"Loading pairs from {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        logger.debug(f"CSV header: {header}")
        pairs = [(row[1].strip(), row[0].strip())          # (neutral, stylised)
                 for row in reader if len(row) >= 2 and row[0].strip() and row[1].strip()]
    if len(pairs) < 2:
        raise ValueError("CSV must contain ≥ 2 rows.")
    logger.info(f"Loaded {len(pairs)} pair(s)")
    return pairs

# Remove _post function
# # ───────────────────────── HTTP helper ─────────────────────────
# def _post(url: str, payload: dict):
# ... (removed) ...

# ───────────────────── GA utilities (roulette etc) ───────────────────────
def _roulette(pop: List["Individual"]) -> "Individual":
    total = sum(max(i.fitness_score, 1e-9) for i in pop)
    pick, acc = random.random() * total, 0.0
    for i in pop:
        acc += max(i.fitness_score, 1e-9)
        if acc >= pick:
            return i
    return pop[-1]

def _parents(pop: List["Individual"]):
    if len(pop) == 1:
        return pop[0], pop[0]
    for _ in range(20):
        a, b = _roulette(pop), _roulette(pop)
        if a.id != b.id:
            return a, b
    # Fallback if roulette fails to find distinct parents quickly
    logger.warning("Roulette failed to find distinct parents after 20 attempts, using random choice.")
    parent_ids = {i.id for i in pop}
    a = random.choice(pop)
    b = random.choice([i for i in pop if i.id != a.id])
    return a, b

@dataclass
class Individual:
    id: int
    generation: int
    prompt: str
    fitness_score: float = 0.0
    feedback: str = ""
    parent_1: int | None = None
    parent_2: int | None = None


# ───────────────────────── GA class (unchanged logic) ────────────────────
class PromptGA:
    def __init__(self, text_category: str, pairs: List[Tuple[str, str]],
                 pop_size=7, generations=3,
                 min_fitness: float | None = None,
                 survival_ratio: float | None = None,
                 num_eval_pairs: int | None = None):
        if min_fitness and survival_ratio:
            raise ValueError("Use min_fitness OR survival_ratio, not both")
        if num_eval_pairs is not None and num_eval_pairs < 1:
             raise ValueError("num_eval_pairs must be >= 1 if specified")
        self.text_category, self.pairs = text_category, pairs
        self.pop_size, self.generations = pop_size, generations
        self.min_fitness, self.survival_ratio = min_fitness, survival_ratio
        self.num_eval_pairs = num_eval_pairs
        self.individuals: List[Individual] = []
        self._next_id = 1

    # — driver —
    def run(self):
        logger.info("=== GA start ===")
        self._initial_pop()
        for g in range(self.generations):
            logger.info(f"-- Generation {g} --")
            self._evaluate(g)
            self._cull(g)
            self._save_csv()                 # live snapshot
            if g < self.generations - 1:
                self._next_gen(g)
        logger.info("=== GA done ===")

    # — steps —
    def _initial_pop(self):
        """Generates the initial population using the Langchain method."""
        logger.info("Generating initial population using Langchain...")
        if not self.pairs:
            raise ValueError("Cannot generate population without loaded pairs.")

        # Use the first pair as examples for generation
        neu0, sty0 = self.pairs[0]

        # Call the function from generate_population module
        prompts = generate_initial_population(
            pop_size=self.pop_size, # Pass pop_size
            text_category=self.text_category,
            neutral_example=neu0,
            stylized_example=sty0
        )

        if not prompts:
            raise RuntimeError("Langchain population generation failed to return any prompts.")

        # --- Log generated prompts for diagnostics ---
        logger.info("--- Raw Initial Prompts Generated ---")
        for i, p_text in enumerate(prompts):
             logger.info(f"Prompt {i+1}: {p_text}")
        logger.info("-------------------------------------")
        # ---------------------------------------------

        if len(prompts) < self.pop_size:
             logger.warning(f"Generated {len(prompts)} prompts, which is less than requested pop_size {self.pop_size}. Proceeding with available prompts.")

        actual_prompts = prompts[:self.pop_size]

        for p in actual_prompts:
            self.individuals.append(Individual(self._next_id, 0, p))
            self._next_id += 1

        logger.info(f"Initialized generation 0 with {len(actual_prompts)} individual(s) using Langchain generator.")

    def _evaluate(self, gen: int):
        if len(self.pairs) < 2:
            logger.warning("Need at least 2 pairs for evaluation, skipping.")
            for ind in [i for i in self.individuals if i.generation == gen]:
                ind.fitness_score = 0.0
                ind.feedback = "Evaluation skipped: requires >= 2 pairs."
            return

        individuals_to_evaluate = [i for i in self.individuals if i.generation == gen]
        logger.info(f"Evaluating {len(individuals_to_evaluate)} individuals for generation {gen}...")

        for ind in individuals_to_evaluate:
            scores, fb = [], []
            
            # Determine evaluation count based on settings
            if self.num_eval_pairs is None:
                eval_count = len(self.pairs)
            else:
                eval_count = min(self.num_eval_pairs, len(self.pairs) - 1)
                eval_count = max(1, eval_count)
            logger.debug(f"Evaluating individual ID {ind.id} with {eval_count} pair(s).")

            for eval_run in range(eval_count):
                # Select pairs for this evaluation run
                if self.num_eval_pairs is None:
                    ex_idx = eval_run
                    tst_idx = (ex_idx + 1) % len(self.pairs)
                else:
                    ex_idx, tst_idx = random.sample(range(len(self.pairs)), 2)

                ex_neu, ex_sty = self.pairs[ex_idx]
                tst_neu, tgt_sty = self.pairs[tst_idx]

                # --- Call Stylizer Chain --- 
                logger.debug(f"Running stylizer for ID {ind.id} (eval {eval_run+1}/{eval_count})")
                stylizer_payload = {
                    "style_prompt": ind.prompt,
                    "neutralized_text_example": ex_neu,
                    "stylized_text_example":    ex_sty,
                    "neutralized_text_to_transform": tst_neu,
                }
                stylized_text = run_stylize_chain(stylizer_payload)

                if not stylized_text:
                     logger.warning(f"Stylizer returned empty result for ID {ind.id}, skipping fitness eval for this run.")
                     continue # Skip fitness if stylizing failed

                # --- Call Fitness Chain --- 
                logger.debug(f"Running fitness eval for ID {ind.id} (eval {eval_run+1}/{eval_count})")
                fitness_payload = {
                    "neutral_text":            tst_neu,
                    "target_stylized_text":    tgt_sty,
                    "stylized_text_to_evaluate": stylized_text,
                }
                fit_result = run_fitness_chain(fitness_payload)

                # Extract score and feedback (handle potential errors/defaults from fitness func)
                score = fit_result.get("fitness_score", 0.0)
                feedback = fit_result.get("feedback", "")

                # Append results for averaging
                scores.append(score)
                fb.append(str(feedback).strip())

            # Calculate and assign average fitness and combined feedback
            if scores:
                ind.fitness_score = sum(scores) / len(scores)
                ind.feedback = " | ".join(fb) # Removed [:500] truncation
            else:
                ind.fitness_score = 0.0 # Assign 0 if all evaluations failed
                ind.feedback = "Evaluation failed for all pairs."

            logger.info(f"ID {ind.id} final fitness {ind.fitness_score:.2f} (from {len(scores)} successful eval(s))")

    def _cull(self, gen: int):
        # (Logic unchanged)
        cur = [i for i in self.individuals if i.generation == gen]
        if not cur or (self.min_fitness is None and self.survival_ratio is None):
            logger.debug(f"Skipping cull for generation {gen} (no individuals or no criteria).")
            return

        if self.min_fitness is not None:
            survivors = [i for i in cur if i.fitness_score >= self.min_fitness]
            logger.debug(f"Culling based on min_fitness >= {self.min_fitness}. {len(survivors)}/{len(cur)} survived.")
        else: # survival_ratio is not None
            keep = max(1, int(round(self.survival_ratio * len(cur))))
            survivors = sorted(cur, key=lambda x: x.fitness_score, reverse=True)[:keep]
            logger.debug(f"Culling based on survival_ratio {self.survival_ratio}. Keeping top {keep}. {len(survivors)}/{len(cur)} survived.")

        survivor_ids = {i.id for i in survivors}
        culled_count = len(cur) - len(survivors)

        if culled_count > 0:
            self.individuals[:] = [i for i in self.individuals if i.generation > gen or i.id in survivor_ids]
            logger.info(f"Culled {culled_count} individual(s) from generation {gen}.")
        else:
             logger.info(f"No individuals culled from generation {gen}.")

    def _next_gen(self, gen: int):
        # Get survivors of the current generation
        survivors = [i for i in self.individuals if i.generation == gen]
        if not survivors:
             logger.error(f"Cannot proceed to next generation: No survivors from generation {gen}. Stopping GA.")
             # Optionally, handle this differently (e.g., raise an error to stop GA earlier)
             # For now, we prevent the error by not proceeding if survivors list is empty.
             # Find a way to terminate the run() loop cleanly.
             # Let's modify run() to check if individuals exist before proceeding.
             return # Exit _next_gen

        logger.info(f"Creating generation {gen+1} from {len(survivors)} survivors...")
        
        # Elitism: Clone the best survivor
        elite = max(survivors, key=lambda x: x.fitness_score)
        self.individuals.append(Individual(self._next_id, gen+1, elite.prompt,
                                           elite.fitness_score, elite.feedback,
                                           elite.id, elite.id))
        logger.debug(f"Cloned elite individual ID {elite.id} (fitness {elite.fitness_score:.2f}) to new ID {self._next_id}")
        self._next_id += 1
        produced = 1
        
        # Crossover: Create remaining individuals
        used_parent_pairs = set()
        max_crossover_attempts = self.pop_size * 5 # Limit attempts to avoid infinite loops
        attempts = 0
        
        while produced < self.pop_size and attempts < max_crossover_attempts:
            attempts += 1
            p1, p2 = _parents(survivors) # Select parents from survivors
            
            # Avoid using the same parent pair multiple times in one generation
            parent_key = tuple(sorted((p1.id, p2.id)))
            if parent_key in used_parent_pairs:
                continue
            used_parent_pairs.add(parent_key)

            logger.debug(f"Running crossover for parents ID {p1.id} and ID {p2.id} -> new individual ID {self._next_id}")
            # --- Call Crossover Chain --- 
            crossover_payload = {
                "neutralized_text_example": self.pairs[0][0],
                "stylized_text_example":    self.pairs[0][1],
                "prompt_1": p1.prompt,
                "prompt_1_fitness_score": p1.fitness_score,
                "prompt_1_fitness_feedback": p1.feedback or "", # Ensure not None
                "prompt_2": p2.prompt,
                "prompt_2_fitness_score": p2.fitness_score,
                "prompt_2_fitness_feedback": p2.feedback or "", # Ensure not None
            }
            child_prompt = run_crossover_chain(crossover_payload)

            if not child_prompt:
                 logger.warning(f"Crossover failed for parents {p1.id}, {p2.id}. Skipping child.")
                 continue # Skip if crossover failed

            self.individuals.append(Individual(
                id=self._next_id,
                generation=gen+1,
                prompt=child_prompt.strip(), # Use the generated child prompt
                parent_1=p1.id,
                parent_2=p2.id
            ))
            self._next_id += 1
            produced += 1

        # Fill remaining slots with elite clones if crossover didn't produce enough
        if produced < self.pop_size:
            logger.warning(f"Crossover produced only {produced-1} children. Filling remaining {self.pop_size - produced} slots with elite clone ID {elite.id}.")
            while produced < self.pop_size:
                self.individuals.append(Individual(self._next_id, gen+1, elite.prompt,
                                                parent_1=elite.id, parent_2=elite.id))
                self._next_id += 1
                produced += 1
        
        logger.info(f"Finished creating generation {gen+1} with {produced} individuals.")


    # live CSV
    def _save_csv(self):
        # (Logic unchanged)
        logger.debug(f"Saving {len(self.individuals)} individuals to {CSV_PATH_OUT}")
        with CSV_PATH_OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID","GEN","PROMPT","FIT","FEEDBACK","P1","P2"])
            for i in self.individuals:
                w.writerow([i.id, i.generation, i.prompt,
                            i.fitness_score, i.feedback,
                            i.parent_1, i.parent_2])
        logger.info("CSV saved")


# ────────────────────────────── run ──────────────────────────────────────
if __name__ == "__main__":
    # Set desired logging level for the full run
    # logger.setLevel(logging.DEBUG) # For max detail
    logger.setLevel(logging.INFO) # For standard info
    # logging.getLogger("generate_population").setLevel(logging.DEBUG) # Uncomment for pop gen detail
    # logging.getLogger("ga_llm_steps").setLevel(logging.DEBUG) # Uncomment for step detail

    logger.info("=== Starting Prompt GA ===")
    try:
        pairs = load_pairs(PAIRS_CSV_IN)
        if not pairs:
             raise RuntimeError("No pairs loaded from CSV, cannot run GA.")

        # Define GA parameters
        ga_text_category = "Udemy lesson scripts for business-related courses"
        ga_pop_size = 7
        ga_generations = 3
        ga_min_fitness = 6.5
        ga_num_eval_pairs = 1 # Or None to use all pairs

        # Initialize and run the GA
        ga = PromptGA(
            text_category=ga_text_category,
            pairs=pairs,
            pop_size=ga_pop_size,
            generations=ga_generations,
            min_fitness=ga_min_fitness,
            # survival_ratio=0.4, # Uncomment if using ratio instead of min_fitness
            num_eval_pairs=ga_num_eval_pairs
        )
        ga.run()

    except Exception as e:
        logger.exception("An error occurred during the GA run.")

    logger.info("=== Prompt GA Finished ===")
