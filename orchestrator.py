#!/usr/bin/env python3
"""
orchestrator.py — Orchestrator.

Routes user messages to specialized agents based on the current task state,
builds context precisely per agent spec, and enforces invariants.

Demo flow:
  [PLAN] > Реализуй функцию parse_csv на Python
      → planner selected (state=PLANNING, model=gpt-4.1-mini)
      → context: profile + invariants + project_memory + state + history(4)

  /state EXEC
  [EXEC] > /step
      → coder selected (state=EXECUTION, model=gpt-4.1)
      → context: profile + invariants + state + history(4) + RULES

  /state VALI
      → validator selected (state=VALIDATION, model=gpt-4.1-mini)
      → context: invariants + state + VALIDATION block (no profile, no history)
"""

import json
import re
import sys

from agents import AgentRegistry, AgentSpec
from context_builder import ContextBuilder


class Orchestrator:
    """
    Selects the appropriate agent for the current task state,
    composes a precise context, calls the LLM, and enforces invariants.

    Wraps Agent (for _post, state, memory) without modifying it.
    """

    def __init__(
        self,
        agent,            # Agent instance (thin caller + state)
        registry: AgentRegistry,
        context_builder: ContextBuilder,
        pricing: dict,
    ):
        """Initialize orchestrator with all collaborators."""
        self.agent = agent
        self.registry = registry
        self.ctx = context_builder
        self.pricing = pricing
        # Pinned agent for the next turn only (set via /agent <name>)
        self._pinned_agent: str | None = None

    # ---------------- Public API ----------------

    def reply(self, user_text: str, agent_name: str | None = None
              ) -> tuple[str, dict]:
        """
        Process a user message: select agent → build context → call LLM.

        Parameters
        ----------
        user_text  : message from the user
        agent_name : explicit agent name override; None = auto-select by state

        Returns
        -------
        (reply_text, metrics_dict)
        metrics includes: model, time, in, out, cost, agent, pre_violations
        """
        ag = self.agent
        tc = ag.tc
        stm = ag.stm
        ltm = ag.ltm

        # Select agent spec
        name = agent_name or self._pinned_agent
        self._pinned_agent = None   # consume pin after one use
        spec = self._select_spec(name, tc.state)

        # Record message in STM
        stm.messages.append({"role": "user", "text": user_text})

        # Facts extraction (when strategy == "facts")
        if ag.context_strategy == "facts":
            _update_facts(ag.facts, user_text)

        # Summary compression before building prompt
        ag._compress_history_if_needed()

        # Pre-check invariants (soft warn)
        pre_violations: list[str] = []
        if ltm.invariants and ltm.checker.rules:
            _, pre_violations = ltm.checker.check(user_text)

        # Build context
        prompt = self.ctx.build(spec, user_text, tc, stm, ltm, ag.facts)

        # Call LLM
        payload = {"model": spec.model, "input": prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        if ag.max_output_tokens is not None:
            payload["max_output_tokens"] = ag.max_output_tokens
        if ag.stop:
            payload["stop"] = ag.stop

        data, elapsed = ag._post(payload)

        if isinstance(data, dict) and data.get("error") is not None:
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error")

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")

        if ag.print_json:
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                text = json.dumps(data, ensure_ascii=False, indent=2)

        # Post-check invariants
        if ltm.invariants and ltm.checker.rules:
            post_passed, post_violations = ltm.checker.check(text)
            if not post_passed:
                if tc.state == "PLANNING":
                    text = _planning_violation_message(post_violations)
                else:
                    text = self._retry_with_violations(
                        text, post_violations, spec
                    )

        stm.messages.append({"role": "assistant", "text": text})

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        _, _, total_cost = _compute_cost(
            self.pricing, spec.model, in_tok, out_tok
        )

        metrics = {
            "model": spec.model,
            "agent": spec.name,
            "time": elapsed,
            "in": in_tok,
            "out": out_tok,
            "cost": total_cost,
            "pre_violations": pre_violations,
        }
        return text, metrics

    def run_step(self) -> tuple[str, dict]:
        """
        Execute the current EXECUTION step via the coder agent.

        Returns (reply_text, metrics_dict).
        Advances tc.step, tc.done, tc.current.
        """
        ag = self.agent
        tc = ag.tc
        stm = ag.stm
        ltm = ag.ltm

        if tc.step >= tc.total:
            return "Все шаги уже выполнены.", _zero_metrics(ag.model)

        current_step = tc.plan[tc.step]
        step_prompt = (
            f"Выполни шаг {tc.step + 1} из {tc.total}: {current_step}\n"
            f"Задача: {tc.task}\n"
            "Выполни только этот шаг. Не переходи к следующим."
        )

        # Use coder spec for execution steps; fall back to state-based default
        spec = self.registry.get("coder") or self._select_spec(None, tc.state)

        stm.messages.append({"role": "user", "text": step_prompt})
        ag._compress_history_if_needed()

        prompt = self.ctx.build(spec, step_prompt, tc, stm, ltm, ag.facts)

        payload = {"model": spec.model, "input": prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        if ag.max_output_tokens is not None:
            payload["max_output_tokens"] = ag.max_output_tokens

        data, elapsed = ag._post(payload)
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error (run_step)")

        try:
            text = data["output"][0]["content"][0]["text"]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        stm.messages.append({"role": "assistant", "text": text})

        # Advance task state
        tc.done.append(current_step)
        tc.step += 1
        tc.current = tc.plan[tc.step] if tc.step < tc.total else ""

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        _, _, total_cost = _compute_cost(self.pricing, spec.model, in_tok, out_tok)

        metrics = {
            "model": spec.model,
            "agent": spec.name,
            "time": elapsed,
            "in": in_tok,
            "out": out_tok,
            "cost": total_cost,
            "pre_violations": [],
        }
        return text, metrics

    def pin_agent(self, name: str) -> str | None:
        """
        Pin an agent by name for the next turn only.
        Returns error string if agent not found, else None.
        """
        if self.registry.get(name) is None:
            return f"Unknown agent '{name}'. Available: {self._agent_names()}"
        self._pinned_agent = name
        return None

    def current_agent_name(self) -> str:
        """Return the name of the agent that would be selected right now."""
        name = self._pinned_agent
        if name:
            return name
        spec = self.registry.for_state(self.agent.tc.state)
        return spec.name if spec else "(none)"

    # ---------------- Internals ----------------

    def _select_spec(self, name: str | None, state: str) -> AgentSpec:
        """
        Select agent spec by name or auto-select by state.
        Falls back to a minimal inline spec if no agent is found.
        """
        if name:
            spec = self.registry.get(name)
            if spec:
                return spec
            print(
                f"[WARN] Agent '{name}' not found, falling back to state-based "
                f"selection",
                file=sys.stderr,
            )

        spec = self.registry.for_state(state)
        if spec:
            return spec

        # Ultimate fallback — bare minimal spec using agent's model
        print(
            f"[WARN] No primary agent for state '{state}', using default model",
            file=sys.stderr,
        )
        return _fallback_spec(self.agent.model, self.agent.system_prompt)

    def _retry_with_violations(
        self, answer: str, violations: list[str], spec: AgentSpec
    ) -> str:
        """Retry LLM once with a correction prompt. Return refusal on second failure."""
        violation_list = "\n".join(f"  - {v}" for v in violations)
        retry_prompt = (
            f"Твой предыдущий ответ нарушает следующие инварианты проекта:\n"
            f"{violation_list}\n\n"
            f"Перепиши ответ, устранив все нарушения. "
            f"Не используй запрещённые инструменты и зависимости.\n\n"
            f"Исходный ответ:\n{answer}"
        )
        payload = {"model": spec.model, "input": retry_prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature

        try:
            data, _ = self.agent._post(payload)
            if isinstance(data, dict) and not data.get("error"):
                retry_text = data["output"][0]["content"][0]["text"]
                passed, retry_violations = self.agent.ltm.checker.check(retry_text)
                if passed:
                    return retry_text
                vlist = "\n".join(f"  - {v}" for v in retry_violations)
                return (
                    f"Не удалось сформировать ответ без нарушений инвариантов.\n"
                    f"Нарушения:\n{vlist}"
                )
        except Exception:
            pass

        vlist = "\n".join(f"  - {v}" for v in violations)
        return (
            f"Ответ нарушает инварианты проекта и не может быть показан.\n"
            f"Нарушения:\n{vlist}"
        )

    def _agent_names(self) -> str:
        """Comma-separated list of all known agent names."""
        return ", ".join(s.name for s in self.registry.list_all())


# ---------------- Module-level helpers ----------------

def _planning_violation_message(violations: list[str]) -> str:
    """Return a correction message for PLANNING state invariant violations."""
    lines = ["Ответ содержит нарушения инвариантов проекта:\n"]
    for i, v in enumerate(violations, 1):
        lines.append(f"  Правило {i}: «{v}»")
    lines.append(
        "\nПожалуйста, переформулируй план без использования "
        "запрещённых инструментов и зависимостей.\n"
        "Ограничения: только requests, venv, без SDK и внешних фреймворков."
    )
    return "\n".join(lines)


def _update_facts(facts: dict, user_text: str):
    """Parse 'Key: Value' lines from user message and update facts store."""
    for line in user_text.splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            key = key.strip()
            value = value.strip()
            if key:
                facts[key] = value


def _compute_cost(pricing: dict, model: str, in_tok, out_tok):
    """Compute cost from token usage and pricing table."""
    from agent import compute_cost
    return compute_cost(pricing, model, in_tok, out_tok)


def _zero_metrics(model: str) -> dict:
    """Return empty metrics dict."""
    return {
        "model": model,
        "agent": "none",
        "time": 0.0,
        "in": 0,
        "out": 0,
        "cost": "$0.000000",
        "pre_violations": [],
    }


def _fallback_spec(model: str, system_prompt: str) -> AgentSpec:
    """Return a minimal AgentSpec when no agent matches the current state."""
    from agents import AgentSpec, ContextPolicy
    return AgentSpec(
        name="fallback",
        mode="primary",
        description="Fallback agent — no spec matched current state",
        model=model,
        temperature=None,
        when_to_use="fallback only",
        allowed_states=["PLANNING", "EXECUTION", "VALIDATION", "DONE"],
        context_policy=ContextPolicy(
            include_profile=True,
            include_invariants=True,
            include_project_memory=True,
            include_state=True,
            include_history=True,
            history_limit=6,
            include_rules_block=True,
            include_validation_block=False,
            include_summary=True,
        ),
        prompt=system_prompt,
    )
