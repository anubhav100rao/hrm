# Self-Healing HR Ops Platform

An agentic HR operations engine (Darwinbox AI Engineering — Advanced track). It reacts to natural-language requests, runs scheduled anomaly scans, consumes upstream system alerts, executes guarded corrective actions, escalates risk to humans (HITL), enforces compliance as hard vetoes, persists episodic memory, and **learns from human feedback** via a reinforcement-learning feedback layer.

## Documentation

- **[architecture-brief.md](architecture-brief.md)** — 1-page architecture brief (the Advanced-only deliverable).
- **[architecture.md](architecture.md)** — full design doc: flow, agents, RL, HITL, compliance, memory, observability, tradeoffs, production path.
- **[interface-contracts.md](interface-contracts.md)** — Pydantic interfaces and example payloads for every node handoff.
- **[llm-instructions.md](llm-instructions.md)** — prompt + structured-output contracts for the LLM-using nodes.

## Architecture diagram

```text
  Reactive (chat/REST)   Scheduled (APScheduler)   System alert (FastAPI webhook)
            \                     |                          /
             +------------ normalize into typed Signal ------+
                                  |
                          LangGraph StateGraph  (SQLite checkpoint)
                                  |
                              Supervisor  (LLM triage / deterministic route)
                                  |
            +---------------------+---------------------+
            v                     v                     v
      Policy Agent          Anomaly Agent         Action request
      (RAG / Chroma)        (pandas/numpy)        (schema validate)
            +---------------------+---------------------+
                                  |
                          Memory Recall (Chroma incidents)
                                  |
                          RL Action Ranker (Thompson bandit)
                                  |
                          Compliance Agent (YAML hard vetoes)
                                  |
                  +---------------+---------------+
                  v                               v
              HITL Gate                      Action Agent
              interrupt() + UI               (mock API tools, idempotent)
                  +---------------+---------------+
                                  |
                   Memory Write -> Reward Update -> Observability
                                  |
                 Final response / tool result / queue item
```
