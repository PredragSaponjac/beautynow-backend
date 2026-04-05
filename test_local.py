"""
Test script: verify Python can talk to LM Studio localhost.
Usage: python test_local.py
"""

import re
import sys
import platform
import subprocess
from openai import OpenAI

from config import BASE_URL, API_KEY, DEFAULT_MAX_TOKENS

_IS_WINDOWS = platform.system() == "Windows"


def check_server():
    """Check if LM Studio server is reachable."""
    print("[TEST] Checking LM Studio server at", BASE_URL, "...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            print("[WARN] Server is up but no models are available.")
            return client, []
        print(f"[OK]   Server is running. Available models: {model_ids}")
        return client, model_ids
    except Exception as e:
        print(f"[FAIL] Cannot reach LM Studio server: {e}")
        print("[HINT] Start LM Studio and enable the local server (port 1234).")
        sys.exit(1)


def check_lms_cli():
    """Check if lms CLI is available."""
    print("[TEST] Checking lms CLI ...")
    try:
        result = subprocess.run(
            ["lms", "status"],
            capture_output=True, text=True, timeout=15,
            shell=_IS_WINDOWS,
        )
        print(f"[OK]   lms CLI works. Output: {result.stdout.strip()[:120]}")
    except FileNotFoundError:
        print("[WARN] lms CLI not found in PATH. Model load/unload will not work.")
        print("[HINT] Make sure LM Studio CLI is installed and in your PATH.")
    except subprocess.TimeoutExpired:
        print("[WARN] lms CLI timed out.")


def pick_test_model(model_ids: list[str]) -> str | None:
    """Pick the best model for testing. Prefer non-thinking models first."""
    # Prefer models that don't have thinking mode issues
    prefer_order = ["phi-4", "qwen2.5-coder", "deepseek", "qwen3"]
    for pref in prefer_order:
        for mid in model_ids:
            if pref in mid.lower() and "embed" not in mid.lower():
                return mid
    # Fallback: first non-embedding model
    for mid in model_ids:
        if "embed" not in mid.lower():
            return mid
    return model_ids[0] if model_ids else None


def test_chat(client, model_id):
    """Send a simple chat message and print the response."""
    print(f"\n[TEST] Sending test message to model: {model_id} ...")

    # Ensure model is loaded
    print("  Loading model via lms CLI ...")
    try:
        subprocess.run(
            ["lms", "load", model_id],
            capture_output=True, timeout=300,
            shell=_IS_WINDOWS,
        )
        import time
        time.sleep(5)
    except Exception:
        print("  [WARN] Could not explicitly load model; trying API call anyway.")

    user_msg = "Say hello and confirm you are running locally. Keep it to one sentence."

    # Disable thinking for models that use it
    if any(tag in model_id.lower() for tag in ["qwen3", "deepseek-r1"]):
        user_msg += "\n\n/no_think"
        print("  [INFO] Thinking-mode model — adding /no_think")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        reply = response.choices[0].message.content or ""
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        print(f"[OK]   Response received ({len(reply)} chars):\n")
        print(reply[:500])
        print()
        return True
    except Exception as e:
        print(f"[FAIL] Chat request failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  LM Studio Local Connection Test")
    print("=" * 60)
    print()

    check_lms_cli()
    print()

    # Unload everything first to ensure clean state
    print("[TEST] Clearing loaded models ...")
    try:
        subprocess.run(
            ["lms", "unload", "--all"],
            capture_output=True, text=True, timeout=60,
            shell=_IS_WINDOWS,
        )
    except Exception:
        pass
    print()

    client, model_ids = check_server()

    if not model_ids:
        print("\n[SKIP] No models available. Download a model in LM Studio first.")
        sys.exit(0)

    model_id = pick_test_model(model_ids)
    if model_id is None:
        print("\n[SKIP] No suitable model found for testing.")
        sys.exit(0)

    print(f"\n[INFO] Selected test model: {model_id}")
    success = test_chat(client, model_id)

    # Clean up
    try:
        subprocess.run(
            ["lms", "unload", "--all"],
            capture_output=True, text=True, timeout=60,
            shell=_IS_WINDOWS,
        )
    except Exception:
        pass

    print("=" * 60)
    if success:
        print("[PASS] All local tests passed. You are ready to run the pipeline.")
    else:
        print("[FAIL] Chat test failed. Check LM Studio logs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
