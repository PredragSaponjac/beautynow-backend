"""
Role-specific prompts for each pipeline stage.
Each function returns a list of messages for the openai chat completion call.
"""


def synthesis_prompt(task: str) -> list[dict]:
    """Stage 1: Second-derivative idea generation."""
    return [
        {
            "role": "system",
            "content": (
                "You are a macro-strategist specializing in second-order and third-order "
                "implications. Your job is NOT to state the obvious consensus trade. "
                "Instead, find ideas that are:\n"
                "- Causally linked to the consensus theme but non-obvious\n"
                "- Less crowded and plausibly under-monetized\n"
                "- Supported by a clear causal chain (A causes B causes C)\n\n"
                "Think like a connect-the-dots analyst. Focus on supply chains, "
                "substitution effects, policy second-effects, and cross-asset spillovers. "
                "Produce 3-5 concrete second-derivative trade ideas with brief causal reasoning."
            ),
        },
        {
            "role": "user",
            "content": task,
        },
    ]


def critic_prompt(task: str, synthesis_output: str) -> list[dict]:
    """Stage 2: Aggressive critique of synthesis ideas."""
    return [
        {
            "role": "system",
            "content": (
                "You are a ruthless devil's-advocate analyst. Your sole job is to "
                "aggressively challenge the ideas presented to you. For each idea:\n"
                "- Explain why it may already be priced in\n"
                "- Identify the weakest assumption in the causal chain\n"
                "- Point out historical precedents where similar logic failed\n"
                "- Rate conviction (1-10) and flag which ideas survive scrutiny\n\n"
                "Do NOT be polite. Be direct, specific, and adversarial. "
                "If an idea is weak, say so plainly and explain why."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL TASK:\n{task}\n\n"
                f"SYNTHESIS OUTPUT (Stage 1):\n{synthesis_output}\n\n"
                "Now critique every idea above. Which ones survive? Which ones are junk?"
            ),
        },
    ]


def verifier_prompt(task: str, synthesis_output: str, critic_output: str) -> list[dict]:
    """Stage 3: Logical verification and structured separation."""
    return [
        {
            "role": "system",
            "content": (
                "You are a logic and verification specialist. Your job is to take the "
                "surviving ideas from a synthesis+critique cycle and produce a rigorous "
                "verified version. Structure your output as:\n\n"
                "1. FACTS — things that are objectively true and verifiable\n"
                "2. ASSUMPTIONS — things assumed but not proven\n"
                "3. INFERENCE — logical deductions from facts + assumptions\n"
                "4. INVALIDATION CONDITIONS — what would kill this thesis\n"
                "5. MONITORING VARIABLES — specific data points to watch\n"
                "6. FINAL VERDICT — the single strongest idea with confidence level\n\n"
                "Be precise. Use numbers where possible. Do not introduce new speculation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL TASK:\n{task}\n\n"
                f"SYNTHESIS (Stage 1):\n{synthesis_output}\n\n"
                f"CRITIQUE (Stage 2):\n{critic_output}\n\n"
                "Now verify: separate facts from assumptions, identify invalidation "
                "conditions, and produce the strongest defensible version."
            ),
        },
    ]


def coder_prompt(task: str, verified_output: str) -> list[dict]:
    """Stage 4: Practical Python code generation."""
    return [
        {
            "role": "system",
            "content": (
                "You are a practical Python developer. Your job is to take a verified "
                "investment thesis and produce a useful Python script that helps monitor "
                "or screen for the identified opportunity. Requirements:\n\n"
                "- Use only standard library + common packages (requests, pandas, etc.)\n"
                "- Include clear comments explaining what each section does\n"
                "- Handle errors gracefully\n"
                "- Print results in a readable format\n"
                "- The script should be immediately runnable\n"
                "- Focus on the monitoring variables identified in the verification stage\n\n"
                "Produce a complete, working Python script — not pseudocode."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL TASK:\n{task}\n\n"
                f"VERIFIED THESIS (Stage 3):\n{verified_output}\n\n"
                "Now produce a Python monitoring/screening script for the strongest idea."
            ),
        },
    ]


# Map stage names to prompt builders
PROMPT_BUILDERS = {
    "synthesis": synthesis_prompt,
    "critic": critic_prompt,
    "verifier": verifier_prompt,
    "coder": coder_prompt,
}
