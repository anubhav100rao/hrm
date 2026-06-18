# Self-Healing HR Ops Platform — Architecture Brief (1 page)

**Problem.** Build the intelligence layer of a Self-Healing HR Ops Platform: an agentic system that reacts to natural-language requests, runs scheduled anomaly scans, consumes upstream system alerts, executes guarded corrective actions, escalates risk to humans, enforces compliance as hard vetoes, remembers past incidents, and learns from human feedback over time.

**Core idea.** Every input — chat/REST request, scheduled scan, or system alert — normalizes into one typed `Signal`. From there the source is irrelevant: a LangGraph `StateGraph` drives the workflow as a sequence of transitions over a shared `GraphState`. Sub-agents only read and write that state; they never call each other directly. A SQLite checkpointer persists state after every node, enabling multi-turn memory, HITL pause/resume, and restart recovery.

**Agents (Supervisor + sub-agents).**
- **Supervisor** — thin LLM triage; classifies reactive requests, routes scheduled/system signals deterministically to anomaly investigation.
- **Policy Agent (RAG)** — Chroma policy index, header-aware chunking, cited answers, "not covered" fallback, cheap citation verifier.
- **Anomaly Agent** — deterministic pandas/numpy detectors (payroll outliers, leave abuse, compliance violations) with calibrated confidence and a 5-action candidate set.
- **Compliance Agent** — 10–15 YAML rules evaluated by a safe expression engine; hard vetoes cannot be overridden by RL, Supervisor, or human.
- **Action Agent** — executes only compliance-approved actions through typed mock APIs with JSON-schema validation, idempotency keys, retries, and audit fallback.
- **Memory nodes** — retrieve/store resolved incidents to warm-start ranking.

**Anomaly → action.** Each anomaly carries evidence, severity, and `confidence = clamp01(signal_strength × data_quality × corroboration_boost)`. The five arms are `auto-correct`, `escalate-to-manager`, `escalate-to-HR`, `flag-for-audit`, `no-action`. High-confidence + low-risk actions auto-proceed after compliance; otherwise they queue for HITL. Confidence and blast radius are separate safety axes.

**RL feedback layer.** A linear Thompson-sampling **contextual bandit** ranks the five actions over a deterministic feature vector (no free-form LLM text). Rewards: HITL approve `+1` / reject `−1` / modify (edit-distance partial) / timeout penalty; delayed outcome (recurrence) reward; false-positive penalty; compliance-veto `−1`. The posterior persists to `data/rl_policy.npz` plus auditable SQLite reward tables, so restarts do not reset learning. The demo shows action-distribution shift across ≥2 feedback cycles.

**HITL.** Triggered when `confidence < AUTO_ACTION_THRESHOLD`, `risk_tier ≥ RISK_THRESHOLD`, or compliance requires approval. The reviewer sees evidence, confidence, citations, RL ranking, compliance findings, and precedents, and can approve/reject/modify (reasons required, fed to RL). Timeout falls back to `flag-for-audit`.

**Memory.** Resolved incidents are embedded in Chroma; similar future anomalies retrieve precedent, warm-start a cloned bandit posterior for that decision, and add a capped corroboration term — so the second occurrence is faster and more confident without one precedent contaminating the global policy.

**Observability & cost.** Every node emits a `StateTransition` (latency, tokens, cost, RL action, rewards) to JSONL + SQLite. A 15-case eval harness covers happy/edge/adversarial/RL scenarios; RL diagnostics plot cumulative reward and action shift; a single LLM client measures a `naive` vs `optimized` mode reporting the required ≥20% token reduction from real output.

**Stack & rationale.** LangGraph (shared-state orchestration that proves "no agent-to-agent calls"), thin provider LLM client (exact token/cost accounting), SQLite + files (zero-ops, inspectable, persistent), Chroma + local embeddings (small offline corpus), FastAPI + Streamlit (typed mock APIs + demo reviewer UI), APScheduler + logical cycles (reproducible before/after RL). Full detail in [architecture.md](architecture.md); contracts in [interface-contracts.md](interface-contracts.md); LLM prompt contracts in [llm-instructions.md](llm-instructions.md).

**Production path.** Swap local infra without changing the learning loop: Kafka/SQS signals, Temporal/LangGraph Platform workflows, warehouse/Spark detectors, hierarchical per-tenant bandit policies, Qdrant/pgvector with PII controls, SLA-aware reviewer queues, a versioned jurisdiction rule registry, and an LLM gateway with per-tenant budgets.
