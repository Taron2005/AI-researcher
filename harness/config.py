"""
Single source of truth for model choices, temperatures, and budget constants.

Every value here is deliberately a plain constant, not a class or a settings
framework — the whole point (CLAUDE.md rule 4) is that changing a model or a
budget number later is a one-line edit here, not a hunt through the codebase.

See DECISIONS.md for *why* each model/temperature was picked, including the
benchmark-verification trail and the two tier-confusion mistakes caught and
corrected along the way.
"""

# --- Model assignment ---
# (CLAUDE.md > "Model assignment" for the reasoning behind each pick)

PLANNER_MODEL = "google/gemini-3.5-flash-lite"
PLANNER_TEMP_DRAFT = 0.6      # blueprint drafting AND candidate-2 design — both real design work
PLANNER_TEMP_JUDGMENT = 0.2   # final write-up only — faithful synthesis, not creative

SWE_MODEL = "deepseek/deepseek-v4-pro"
SWE_TEMP = 0.2                # code generation benefits from low randomness

REVIEWER_MODEL = "openai/gpt-5-mini"
REVIEWER_TEMP = 0.2           # consistent, strict judgment

# --- Self-imposed research budget ---
# Tracked here and enforced by the orchestrator's own code, never left for an
# LLM to remember on its own (Deep Thought failure #6).

MAX_CANDIDATES = 2             # cheap baseline, then one 3D-aware GNN (DECISIONS.md)
MAX_REVIEW_FIX_ROUNDS = 5      # hard retry cap per candidate (failure #7)

# --- Per-role tool-loop turn caps ---
# Not all roles need the same session length -- the Planner's research turns
# are naturally shorter than the Software Engineer writing several files.
PLANNER_MAX_TOOL_TURNS = 8
SWE_MAX_TOOL_TURNS = 20   # was 15; raised after a real run showed ~18.7k prompt
                          # tokens by turn 15 (DECISIONS.md) -- deliberately modest,
                          # since simply raising this further risks running into the
                          # documented ~20-50k tool-calling degradation threshold
                          # rather than avoiding it. The actual fix was scoping the
                          # prompt tighter (no hyperparameter tuning during self-check),
                          # not just adding more turns.

# --- OpenRouter endpoint ---
# The API key itself is loaded from .env by llm_client.py, not here — this
# file has zero side effects and can be imported without a working .env.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
