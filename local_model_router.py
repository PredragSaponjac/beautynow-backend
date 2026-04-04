"""
Local 4-model orchestration pipeline.

Runs a task through 4 stages, each using a different local LM Studio model:
  1. SYNTHESIS  — qwen/qwen3-14b
  2. CRITIQUE   — deepseek/deepseek-r1-0528-qwen3-8b
  3. VERIFY     — microsoft/phi-4-reasoning-plus
  4. CODE       — qwen/qwen2.5-coder-14b

Usage:
  python local_model_router.py                     # uses default task
  python local_model_router.py "Your custom task"  # custom task
"""

import os
import sys
import time
import platform
import subprocess
from datetime import datetime

from openai import OpenAI

from config import (
    BASE_URL, API_KEY, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    LOAD_WAIT_SECONDS, LMS_TIMEOUT_SECONDS, MODELS, STAGES,
    OUTPUT_DIR, DEFAULT_TASK,
)
from prompts import PROMPT_BUILDERS


# On Windows, lms is often a .cmd shim — subprocess needs shell=True to find it
_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Model management via lms CLI
# ---------------------------------------------------------------------------

def lms_run(command: list[str], label: str, timeout: int = LMS_TIMEOUT_SECONDS) -> bool:
    """Run an lms CLI command. Returns True on success."""
    cmd = ["lms"] + command
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, shell=_IS_WINDOWS,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout
            print(f"  [WARN] {label} returned code {result.returncode}: {detail[:200]}")
            return False
        return True
    except FileNotFoundError:
        print(f"  [ERROR] lms CLI not found. Cannot {label}.")
        print(f"  [HINT]  Make sure LM Studio CLI is in your PATH.")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] lms CLI timed out ({timeout}s) during {label}.")
        return False


def unload_all():
    """Unload all currently loaded models."""
    print("[UNLOAD] Unloading all models ...")
    # Try --all first; if it fails, fall back to unloading without args
    if not lms_run(["unload", "--all"], "unload --all", timeout=60):
        print("  [INFO] --all flag may not be supported; trying plain unload ...")
        lms_run(["unload"], "unload", timeout=60)


def load_model(model_id: str) -> bool:
    """Load a specific model into LM Studio."""
    print(f"[LOAD] Loading {model_id} ...")
    success = lms_run(["load", model_id], f"load {model_id}")
    if success:
        print(f"  Waiting {LOAD_WAIT_SECONDS}s for model to initialize ...")
        time.sleep(LOAD_WAIT_SECONDS)
    return success


# ---------------------------------------------------------------------------
# Detect loaded model ID from the server
# ---------------------------------------------------------------------------

def get_loaded_model_id(client: OpenAI, expected_fragment: str) -> str | None:
    """
    Ask the LM Studio server which models are loaded and return the one
    whose ID contains the expected_fragment (e.g. 'qwen3-14b').
    Falls back to the first loaded model if no match.
    LM Studio may report model IDs differently from lms CLI names.
    """
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            return None
        # Try to match by fragment (e.g. "qwen3-14b" in "qwen-qwen3-14b-GGUF")
        for mid in model_ids:
            if expected_fragment.lower() in mid.lower():
                return mid
        # No exact match — return first loaded model
        print(f"  [INFO] Expected '{expected_fragment}' but server reports: {model_ids}")
        print(f"  [INFO] Using first loaded model: {model_ids[0]}")
        return model_ids[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(client: OpenAI, model_id: str, messages: list[dict]) -> str:
    """Send a chat completion request to the loaded model."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Model call failed: {e}"


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_output(stage_name: str, content: str, run_id: str) -> str:
    """Save stage output to a timestamped file. Returns the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{run_id}_{stage_name}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(task: str):
    """Execute the full 4-stage pipeline."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Verify server is reachable
    print("\n[CHECK] Verifying LM Studio server ...")
    try:
        client.models.list()
        print("[OK]    Server is reachable.\n")
    except Exception as e:
        print(f"[FAIL]  Cannot reach LM Studio at {BASE_URL}: {e}")
        print("[HINT]  Start LM Studio and enable the local server (port 1234).")
        sys.exit(1)

    print("=" * 70)
    print("  LOCAL MULTI-MODEL PIPELINE")
    print(f"  Run ID : {run_id}")
    print(f"  Task   : {task[:80]}{'...' if len(task) > 80 else ''}")
    print("=" * 70)

    # Collect outputs from each stage to feed into the next
    stage_outputs = {}

    for stage_key, stage_label in STAGES:
        config_model_id = MODELS[stage_key]

        print(f"\n{'—' * 70}")
        print(f"[{stage_label}]")
        print(f"  Model: {config_model_id}")
        print(f"{'—' * 70}")

        # Unload previous, load current
        unload_all()
        if not load_model(config_model_id):
            print(f"  [ERROR] Failed to load {config_model_id}. Skipping stage.")
            stage_outputs[stage_key] = f"[SKIPPED — failed to load {config_model_id}]"
            continue

        # Resolve actual model ID from the server (may differ from config name)
        # Use last part of model path as the match fragment
        fragment = config_model_id.split("/")[-1]
        actual_model_id = get_loaded_model_id(client, fragment)
        if actual_model_id is None:
            print(f"  [ERROR] No model appears loaded on the server after lms load.")
            stage_outputs[stage_key] = f"[SKIPPED — model not detected on server]"
            continue
        if actual_model_id != config_model_id:
            print(f"  [INFO] Server reports model as: {actual_model_id}")

        # Build prompt based on stage
        if stage_key == "synthesis":
            messages = PROMPT_BUILDERS[stage_key](task)
        elif stage_key == "critic":
            messages = PROMPT_BUILDERS[stage_key](task, stage_outputs.get("synthesis", ""))
        elif stage_key == "verifier":
            messages = PROMPT_BUILDERS[stage_key](
                task,
                stage_outputs.get("synthesis", ""),
                stage_outputs.get("critic", ""),
            )
        elif stage_key == "coder":
            messages = PROMPT_BUILDERS[stage_key](task, stage_outputs.get("verifier", ""))

        # Call model
        print("  Calling model ...")
        output = call_model(client, actual_model_id, messages)
        stage_outputs[stage_key] = output

        # Save output
        filepath = save_output(stage_key, output, run_id)
        print(f"  Saved  : {filepath}")

        # Preview
        preview = output[:300].replace("\n", "\n  ")
        print(f"\n  --- Preview ---\n  {preview}")
        if len(output) > 300:
            print(f"  ... ({len(output)} chars total)")

    # Final cleanup
    print(f"\n{'—' * 70}")
    print("[UNLOAD] Final cleanup ...")
    unload_all()

    # Save combined output
    combined = ""
    for stage_key, stage_label in STAGES:
        combined += f"{'=' * 70}\n{stage_label}\n{'=' * 70}\n"
        combined += stage_outputs.get(stage_key, "[NO OUTPUT]") + "\n\n"

    combined_path = save_output("COMBINED", combined, run_id)

    # Summary
    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Run ID         : {run_id}")
    print(f"  Combined output: {combined_path}")
    print(f"  Output dir     : {OUTPUT_DIR}/")
    print(f"  Files created  :")
    for stage_key, _ in STAGES:
        print(f"    - {run_id}_{stage_key}.txt")
    print(f"    - {run_id}_COMBINED.txt")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = DEFAULT_TASK
        print("[INFO] No task provided. Using default sample task.\n")

    run_pipeline(task)


if __name__ == "__main__":
    main()
