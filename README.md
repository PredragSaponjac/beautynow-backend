# Local LM Studio Multi-Model Orchestration

A fully local 4-model pipeline that runs on your desktop via LM Studio. No cloud APIs, no Docker, no web apps — just Python scripts and your local GPU.

## Architecture

```
Task Input
    │
    ▼
┌──────────────────────────────┐
│ STEP 1 — SYNTHESIS           │  qwen/qwen3-14b
│ Second-derivative ideas      │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 2 — CRITIQUE            │  deepseek/deepseek-r1-0528-qwen3-8b
│ Challenge assumptions        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 3 — VERIFY              │  microsoft/phi-4-reasoning-plus
│ Facts vs assumptions         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 4 — CODE                │  qwen/qwen2.5-coder-14b
│ Python monitoring script     │
└──────────────────────────────┘
           │
           ▼
      outputs/ folder
```

## Prerequisites

1. **LM Studio** installed and running with the local server enabled on port 1234
2. **Python 3.10+** installed and in PATH
3. **openai** Python package: `pip install openai`
4. **lms CLI** available in PATH (comes with LM Studio)
5. These models downloaded in LM Studio:
   - `qwen/qwen3-14b`
   - `deepseek/deepseek-r1-0528-qwen3-8b`
   - `microsoft/phi-4-reasoning-plus`
   - `qwen/qwen2.5-coder-14b`

## Quick Start

### 1. Test your connection

```bash
python test_local.py
```

This checks:
- LM Studio server is reachable at `localhost:1234`
- `lms` CLI works
- A loaded model can respond to a chat message

### 2. Run the full pipeline

**Using the batch file (Windows):**
```bash
run_pipeline.bat
```

**Using Python directly:**
```bash
python local_model_router.py
```

**With a custom task:**
```bash
python local_model_router.py "Your custom analysis task here"
```

### 3. Check outputs

All stage outputs are saved to `outputs/` with timestamps:
```
outputs/
  20260404_143022_synthesis.txt
  20260404_143022_critic.txt
  20260404_143022_verifier.txt
  20260404_143022_coder.txt
  20260404_143022_COMBINED.txt
```

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Model IDs, server URL, generation settings |
| `prompts.py` | Role-specific system/user prompts for each stage |
| `test_local.py` | Connection test script |
| `local_model_router.py` | Main pipeline orchestrator |
| `run_pipeline.bat` | Windows batch launcher |
| `outputs/` | Timestamped output files |

## How It Works

1. The pipeline unloads any currently loaded model
2. Loads the model needed for the current stage
3. Waits a few seconds for the model to initialize
4. Sends the stage-specific prompt (including outputs from prior stages)
5. Saves the response to a timestamped file
6. Repeats for all 4 stages
7. Saves a combined output file

Only one large model is loaded at a time to manage VRAM.

## Configuration

Edit `config.py` to change:
- `LOAD_WAIT_SECONDS` — pause after loading a model (default: 5s)
- `DEFAULT_MAX_TOKENS` — max output tokens per stage (default: 2048)
- `DEFAULT_TEMPERATURE` — generation temperature (default: 0.7)
- `MODELS` — swap in different model IDs

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot reach LM Studio` | Start LM Studio, enable local server on port 1234 |
| `lms CLI not found` | Install LM Studio CLI or add it to PATH |
| `No models loaded` | Run `lms load qwen/qwen3-14b` manually first |
| `Model load fails` | Check the model is downloaded in LM Studio |
| Slow responses | Normal for 14B models on consumer hardware |

## v2 Upgrade Path

Recommended next improvement: **Add a feedback loop** — let the critic re-evaluate the coder's output and iterate until the code passes a quality bar. This turns the linear pipeline into a refinement cycle.
