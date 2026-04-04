"""
Configuration for local LM Studio multi-model orchestration.
All models run locally via LM Studio at localhost:1234.
"""

# LM Studio local server
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio-local"  # dummy key, required by openai client

# Default generation settings
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# Pause (seconds) after loading a model before sending requests
LOAD_WAIT_SECONDS = 8

# Timeout (seconds) for lms CLI commands (loading 14B models can be slow)
LMS_TIMEOUT_SECONDS = 300

# Model identifiers — must match what LM Studio has downloaded
MODELS = {
    "synthesis": "qwen/qwen3-14b",
    "critic":    "deepseek/deepseek-r1-0528-qwen3-8b",
    "verifier":  "microsoft/phi-4-reasoning-plus",
    "coder":     "qwen/qwen2.5-coder-14b",
}

# Pipeline stage labels
STAGES = [
    ("synthesis", "STEP 1 — IDEA / SYNTHESIS"),
    ("critic",    "STEP 2 — CRITIQUE"),
    ("verifier",  "STEP 3 — VERIFIED VERSION"),
    ("coder",     "STEP 4 — CODE OUTPUT"),
]

# Output directory
OUTPUT_DIR = "outputs"

# Default sample task
DEFAULT_TASK = (
    "Consensus trade: everyone is buying oil. "
    "Do not give the first-order trade. "
    "Find second-derivative ideas that are causally linked, less crowded, "
    "and plausibly under-monetized. "
    "Then critique them, verify the strongest version, "
    "and produce a Python monitoring or screening script."
)
