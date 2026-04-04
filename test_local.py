"""
Test script: verify Python can talk to LM Studio localhost.
Usage: python test_local.py
"""

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
            print("[WARN] Server is up but no models are loaded.")
            print("[HINT] Load a model with: lms load <model-name>")
            return client, None
        print(f"[OK]   Server is running. Loaded models: {model_ids}")
        return client, model_ids[0]
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


def test_chat(client, model_id):
    """Send a simple chat message and print the response."""
    print(f"\n[TEST] Sending test message to model: {model_id} ...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                {"role": "user", "content": "Say hello and confirm you are running locally."},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
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

    client, model_id = check_server()

    if model_id is None:
        print("\n[SKIP] No model loaded — cannot test chat. Load a model first.")
        print("       Example: lms load qwen/qwen3-14b")
        sys.exit(0)

    success = test_chat(client, model_id)

    print("=" * 60)
    if success:
        print("[PASS] All local tests passed. You are ready to run the pipeline.")
    else:
        print("[FAIL] Chat test failed. Check LM Studio logs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
