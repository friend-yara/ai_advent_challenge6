#!/usr/bin/env python3
"""
context_builder.py — ContextBuilder.

Assembles the full prompt string for an LLM call based on an AgentSpec's
context_policy. Replaces Agent._build_prompt from the monolithic agent.

Block injection order:
  SYSTEM → PROFILE → PROJECT_MEMORY → INVARIANTS → STATE → RULES
  → VALIDATION → SUMMARY → FACTS → DIALOG
"""

try:
    import toons
except ImportError as e:
    raise SystemExit("ERROR: Install TOON support: pip install toons") from e

from agents import AgentSpec, ContextPolicy


class ContextBuilder:
    """
    Builds a prompt string by selectively injecting context layers
    according to the AgentSpec's context_policy.

    All data is passed in at call time — no internal state.
    """

    def build(
        self,
        spec: AgentSpec,
        user_text: str,
        tc,           # TaskContext
        stm,          # ShortTermMemory
        ltm,          # LongTermMemory
        facts: dict,
    ) -> str:
        """
        Assemble the full prompt for an LLM call.

        Parameters
        ----------
        spec      : AgentSpec — drives which blocks to include
        user_text : the current user message
        tc        : TaskContext — working memory / state
        stm       : ShortTermMemory — dialogue history + summary
        ltm       : LongTermMemory — profile, invariants, project memory
        facts     : key/value facts store

        Returns
        -------
        Full prompt string ready to send as the `input` field.
        """
        cp: ContextPolicy = spec.context_policy
        parts: list[str] = []

        # SYSTEM — always injected (agent's own system prompt)
        parts.append("SYSTEM:\n" + spec.prompt.strip())

        # PROFILE
        if cp.include_profile and ltm.use_profile:
            if ltm.profile_obj and ltm.profile_obj.enabled:
                label = ltm.profile_obj.inject_as
                parts.append(f"\n{label}:\n{ltm.profile_obj.to_yaml()}")

        # PROJECT_MEMORY
        if cp.include_project_memory and ltm.use_project_memory:
            if ltm.project_memory:
                parts.append(f"\nPROJECT_MEMORY:\n{ltm.project_memory}")

        # INVARIANTS
        if cp.include_invariants and ltm.use_invariants:
            if ltm.invariants:
                parts.append(f"\nINVARIANTS:\n{ltm.invariants}")

        # STATE — full TaskContext dump, or task-only, or plan summary
        if cp.include_state:
            parts.append(
                "\nSTATE (TOON v3.0):\n" + toons.dumps(tc.to_dict()).strip()
            )
        elif cp.include_task and tc.task:
            parts.append(f"\nTASK:\n{tc.task}")
        elif cp.include_plan_summary:
            _add_plan_summary(parts, tc)

        # RULES — only meaningful in EXECUTION when there is a current step
        if cp.include_rules_block and tc.current:
            parts.append(
                "\nRULES:\n"
                f"- Work only within the current step: {tc.current}\n"
                "- Do not skip steps\n"
                "- When step is complete, signal completion explicitly"
            )

        # VALIDATION — only in VALIDATION state
        if cp.include_validation_block and ltm.invariants:
            parts.append(
                "\nVALIDATION:\n"
                "Check the completed steps and this answer against INVARIANTS. "
                "Report any violations explicitly with the rule text and "
                "a suggested fix."
            )

        # SUMMARY — compressed dialogue summary
        if cp.include_summary and stm.summary:
            parts.append(f"\nSUMMARY:\n{stm.summary.strip()}")

        # FACTS
        if cp.include_facts and facts:
            facts_lines = "\n".join(f"{k}={v}" for k, v in facts.items())
            parts.append(f"\nFACTS:\n{facts_lines}")

        # DIALOG — sliding window over recent messages
        limit = cp.history_limit if cp.include_history else 0
        if limit > 0:
            recent = stm.messages[-limit:]
            if recent:
                parts.append("\nDIALOG:")
                for m in recent:
                    role = m["role"]
                    text = m["text"]
                    if role == "user":
                        parts.append(f"User: {text}")
                    else:
                        parts.append(f"Assistant: {text}")

        parts.append("Assistant:")
        return "\n".join(parts)


# ---------------- Helpers ----------------

def _add_plan_summary(parts: list[str], tc) -> None:
    """Inject a compact plan+done summary block (for reviewer agent)."""
    lines = []
    if tc.task:
        lines.append(f"Task: {tc.task}")
    if tc.done:
        lines.append("Completed steps:")
        for step in tc.done:
            lines.append(f"  [x] {step}")
    if tc.plan:
        remaining = tc.plan[tc.step:]
        if remaining:
            lines.append("Remaining steps:")
            for step in remaining:
                lines.append(f"  [ ] {step}")
    if lines:
        parts.append("\nPLAN_SUMMARY:\n" + "\n".join(lines))
