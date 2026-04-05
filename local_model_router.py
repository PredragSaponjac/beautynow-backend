"""
Local 4-model orchestration pipeline.

Runs a task through 4 stages, each using a different local LM Studio model:
  1. SYNTHESIS  — qwen/qwen3-14b
  2. CRITIQUE   — deepseek/deepseek-r1-0528-qwen3-8b
  3. VERIFY     — microsoft/phi-4-reasoning-plus
  4. CODE       — qwen/qwen2.5-coder-14b

LM Studio advertises all downloaded models via the API and loads them
on-demand when a request arrives. Between stages we unload via lms CLI
to free VRAM so only one large model is active at a time.

Usage:
  python local_model_router.py                     # uses default task
  python local_model_router.py "Your custom task"  # custom task
"""

import os
import re
import sys
import time
import platform
import subprocess
from datetime import datetime

from openai import OpenAI

from config import (
    BASE_URL, API_KEY, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    LOAD_WAIT_SECONDS, LMS_TIMEOUT_SECONDS, HTTP_TIMEOUT_SECONDS,
    MAX_RETRIES, RETRY_WAIT_SECONDS, MODELS, STAGES,
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
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout
            print(f"  [WARN] {label} returned code {result.returncode}: {detail[:200]}")
            return False
        stdout = (result.stdout or "").strip()
        if stdout:
            print(f"  {stdout[:150]}")
        return True
    except FileNotFoundError:
        print(f"  [ERROR] lms CLI not found. Cannot {label}.")
        print(f"  [HINT]  Make sure LM Studio CLI is in your PATH.")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] lms CLI timed out ({timeout}s) during {label}.")
        return False


def unload_all():
    """Unload all currently loaded models to free VRAM."""
    print("[UNLOAD] Freeing VRAM ...")
    if not lms_run(["unload", "--all"], "unload --all", timeout=60):
        lms_run(["unload"], "unload", timeout=60)
    # Give GPU memory time to fully release
    print("  Waiting 5s for VRAM to clear ...")
    time.sleep(5)


def ensure_model_loaded(model_id: str):
    """
    Use lms CLI to explicitly load a model.
    Uses --gpu 0 to force CPU-only loading (this machine has only 2GB VRAM).
    """
    print(f"[LOAD] Ensuring {model_id} is ready ...")
    # Try with --gpu 0 first (CPU-only), fall back to default if flag not supported
    if not lms_run(["load", model_id, "--gpu", "0"], f"load {model_id}"):
        print("  [INFO] Retrying without --gpu flag ...")
        lms_run(["load", model_id], f"load {model_id}")
    print(f"  Waiting {LOAD_WAIT_SECONDS}s for model to initialize ...")
    time.sleep(LOAD_WAIT_SECONDS)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(client: OpenAI, model_id: str, messages: list[dict]) -> str:
    """Send a chat completion request to the model, with retry on transient errors."""
    patched_messages = _patch_thinking_models(model_id, messages)

    for attempt in range(1, MAX_RETRIES + 2):  # +2 because range is exclusive and attempt 1 is first try
        try:
            if attempt > 1:
                print(f"  Retry {attempt - 1}/{MAX_RETRIES} after {RETRY_WAIT_SECONDS}s ...")
                time.sleep(RETRY_WAIT_SECONDS)

            print(f"  Sending request to {model_id} (max_tokens={DEFAULT_MAX_TOKENS}, "
                  f"timeout={HTTP_TIMEOUT_SECONDS}s) ...")
            response = client.chat.completions.create(
                model=model_id,
                messages=patched_messages,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content or ""
            content = _strip_think_tags(content)
            return content.strip()
        except Exception as e:
            error_str = str(e)
            # Retry on transient errors
            is_transient = any(msg in error_str for msg in [
                "Model reloaded", "timed out", "Connection error",
                "Failed to load model",
            ])
            if is_transient and attempt <= MAX_RETRIES:
                print(f"  [WARN] Transient error: {error_str[:120]}")
                continue
            return f"[ERROR] Model call failed: {e}"


def _patch_thinking_models(model_id: str, messages: list[dict]) -> list[dict]:
    """
    Only disable thinking for Qwen3 models (which crash without /no_think).
    DeepSeek-R1 handles thinking internally — we just strip <think> tags
    from the output instead.
    """
    if "qwen3" not in model_id.lower() or "deepseek" in model_id.lower():
        return messages

    print(f"  [INFO] Qwen3 model detected — adding /no_think")
    patched = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1 and msg.get("role") == "user":
            patched.append({**msg, "content": msg["content"] + "\n\n/no_think"})
        else:
            patched.append(msg)
    return patched


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"[OK]    Server is reachable. Available models: {len(available)}")

        # Verify our required models are available
        for stage_key, _ in STAGES:
            mid = MODELS[stage_key]
            if mid not in available:
                print(f"[WARN]  Model '{mid}' not found in LM Studio. Stage may fail.")
        print()
    except Exception as e:
        print(f"[FAIL]  Cannot reach LM Studio at {BASE_URL}: {e}")
        print("[HINT]  Start LM Studio and enable the local server (port 1234).")
        sys.exit(1)

    print("=" * 70)
    print("  LOCAL MULTI-MODEL PIPELINE")
    print(f"  Run ID : {run_id}")
    print(f"  Task   : {task[:80]}{'...' if len(task) > 80 else ''}")
    print("=" * 70)

    # Start clean — unload anything in memory
    unload_all()

    # Collect outputs from each stage to feed into the next
    stage_outputs = {}
    last_loaded_model = None

    for stage_key, stage_label in STAGES:
        model_id = MODELS[stage_key]

        print(f"\n{'—' * 70}")
        print(f"[{stage_label}]")
        print(f"  Model: {model_id}")
        print(f"{'—' * 70}")

        # Only reload if model changed from previous stage
        if model_id != last_loaded_model:
            unload_all()
            ensure_model_loaded(model_id)
            last_loaded_model = model_id
        else:
            print("  [INFO] Same model — skipping reload")

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
        output = call_model(client, model_id, messages)

        # Check for errors
        if output.startswith("[ERROR]"):
            print(f"  {output}")
            stage_outputs[stage_key] = output
        else:
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

def load_report(filepath: str) -> str:
    """Load a research report from a text file."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Report file not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    char_count = len(content)
    word_count = len(content.split())
    page_est = word_count // 300  # rough estimate

    print(f"[REPORT] Loaded: {filepath}")
    print(f"  Words: {word_count:,} | Chars: {char_count:,} | ~{page_est} pages")

    # Warn if report is very long (may exceed model context window)
    if word_count > 8000:
        print(f"  [WARN] Report is long ({word_count:,} words). "
              f"Models may truncate input.")
        print(f"  [HINT] For 50+ page reports, consider summarizing first.")

    return content


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Local 4-model orchestration pipeline via LM Studio"
    )
    parser.add_argument(
        "task", nargs="?", default=None,
        help="The analysis task (text). Ignored if --report is used."
    )
    parser.add_argument(
        "--report", "-r", type=str, default=None,
        help="Path to a research report file (.txt, .md). "
             "The report content is prepended to the task as context."
    )
    parser.add_argument(
        "--task-prompt", "-t", type=str, default=None,
        help="Custom task/question to ask about the report. "
             "Defaults to the standard second-derivative analysis prompt."
    )
    args = parser.parse_args()

    # Build the task string
    if args.report:
        report_content = load_report(args.report)
        question = args.task_prompt or args.task or DEFAULT_TASK

        task = (
            f"=== RESEARCH REPORT ===\n\n"
            f"{report_content}\n\n"
            f"=== END OF REPORT ===\n\n"
            f"=== YOUR TASK ===\n\n"
            f"Based on the research report above, {question}"
        )
        print(f"[TASK]   {question[:80]}{'...' if len(question) > 80 else ''}\n")
    elif args.task:
        task = args.task
    else:
        task = DEFAULT_TASK
        print("[INFO] No task provided. Using default sample task.\n")

    run_pipeline(task)


if __name__ == "__main__":
    main()
