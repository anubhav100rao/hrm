# Self-Healing HR Ops Platform - Architecture Brief

## 1. Scope

This design targets the Darwinbox advanced assignment: an agentic HR operations engine that can react to natural-language requests, run scheduled anomaly scans, consume upstream system alerts, execute guarded corrective actions, and learn from human feedback over time.

The implementation is a single Python 3.11+ repo with generated mock artifacts: 500-1,000 employees, 3-5 pages of HR policy documents, seeded payroll/attendance/leave/performance data, and a FastAPI mock API spec. The architecture is local-first for the demo, but each storage and orchestration choice has a clear production replacement.

## 2. Requirement Mapping

| Requirement | Design choice |
|---|---|
| Supervisor plus at least 4 sub-agents | LangGraph `StateGraph` with Supervisor, Policy Agent, Anomaly Detection Agent, Compliance Agent, Action Agent, plus Memory nodes |
| No direct agent-to-agent calls | Agents only read and write shared `GraphState`; conditional graph edges route all transitions |
| Three trigger classes | Reactive REST/chat requests, APScheduler scans, and FastAPI alert webhooks all normalize into one `Signal` envelope |
| RAG grounded in HR policies | Chroma policy index, header-aware chunking, cited answers, and "not covered" fallback for missing policy support |
| Anomaly detection with confidence and action | Deterministic statistics/rules produce calibrated confidence and a 5-action candidate set |
| RL feedback layer | Contextual bandit ranks the 5 actions and updates from HITL, outcomes, false positives, and compliance vetoes |
| HITL approval | LangGraph `interrupt()` pauses high-risk or low-confidence decisions; FastAPI/Streamlit reviewer UI resumes runs |
| Compliance rules | 10-15 YAML rules evaluated by a pure Python rules engine; hard vetoes cannot be overridden |
| Episodic memory | Chroma stores resolved incidents and warm-starts action ranking for similar future cases |
| Observability and evals | Structured traces, 15-case evaluation harness, RL diagnostics, token/cost ledger, and cost comparison mode |

## 3. High-Level Flow

```text
Reactive request       Scheduled scan       System alert
      |                    |                    |
      +--------- normalize into Signal --------+
                           |
                           v
                    LangGraph StateGraph
                           |
                    +------+------+
                    | Supervisor |
                    +------+------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
  Policy Agent      Anomaly Agent       Compliance Agent
     (RAG)          (stats/rules)        (YAML veto)
        |                  |                  |
        +------------------+------------------+
                           |
                    Memory Recall
                           |
                    RL Action Ranker
                           |
                    Compliance Check
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
   HITL Gate if needed              Action Agent if safe
          |                                 |
          +---------------+-----------------+
                          |
                   Memory Write
                          |
                   Reward Update
                          |
                Trace, metrics, response
```

All graph nodes append a `StateTransition` to shared state: node name, input digest, output summary, latency, tool calls, token usage, cost, selected RL action, and rewards. The same event is written to JSONL and indexed in SQLite for the live trace UI.

## 4. Core State and Routing

Every trigger becomes a typed `Signal`:

```python
class Signal(BaseModel):
    signal_id: str
    source: Literal["reactive", "scheduled", "system"]
    received_at: datetime
    raw_payload: dict
    employee_ids: list[str] = []
    priority: Literal["low", "normal", "high"] = "normal"
```

The graph state contains the signal, extracted intent, retrieved policy chunks, detected anomalies, proposed action, compliance result, HITL decision, tool results, RL context vector, selected arm, reward events, and transition log.

The LangGraph checkpointer is keyed by `thread_id`, so multi-turn reactive sessions and paused HITL runs survive process restarts. The Supervisor is intentionally thin. Scheduled and system-generated signals route deterministically to anomaly investigation. Reactive requests use a small LLM call with structured output to classify intent as policy lookup, action request, anomaly report, or mixed. Mixed requests fan out through the graph, but agents never call each other directly.

## 5. Agents

**Policy Agent:** Retrieves from the HR policy corpus using Chroma. Policies are split by markdown headings and clauses so numeric limits stay with their conditions. The agent returns cited answers only; if retrieved chunks do not support an answer, it says the policy corpus does not cover the question. A cheap verifier checks that claims are entailed by cited chunks.

**Anomaly Detection Agent:** Uses deterministic pandas/numpy detectors, not LLM judgment. Payroll outliers use robust z-scores within peer cohorts such as department, band, and country. Leave abuse uses clustering around weekends/holidays, policy-limit pressure, and threshold-gaming patterns. Compliance violations include missing training, overtime cap breaches, probation constraints, and correction-window issues.

**Compliance Agent:** Evaluates every proposed action against YAML rules before execution. Rules include overtime caps, leave notice periods, probation restrictions, payroll correction tiers, mandatory training gates, retro-correction windows, and double-correction locks. A hard veto blocks the action even if RL, the Supervisor, or a human reviewer preferred it.

**Action Agent:** Executes only compliance-approved actions through mock API tools: payroll adjustment, leave flag, ticket escalation, audit log, and training assignment. Guardrails include JSON-schema validation, idempotency keys, parameter re-checks, retries for transient failures, and dry-run mode for evals.

**Memory Nodes:** Retrieve and write resolved incidents. Memory is not used as untrusted free text for tools; it biases ranking and helps humans see precedents.

## 6. Confidence and Action Selection

Each detected anomaly carries:

```python
class Anomaly(BaseModel):
    anomaly_id: str
    employee_id: str
    anomaly_type: Literal["payroll", "leave", "compliance"]
    evidence: list[EvidenceItem]
    confidence: float
    severity: float
    recommended_action: ActionCandidate
    action_candidates: list[ActionCandidate]
```

Confidence is calibrated against seeded ground-truth anomalies:

```text
confidence = calibrated_signal_strength * data_quality * corroboration_boost
```

Signal strength comes from detector-specific statistics, such as robust z-score magnitude or tail probability. Data quality penalizes small cohorts, missing fields, and legitimate context changes. Corroboration increases confidence when independent signals agree, for example payroll delta plus unmatched overtime data.

The five action arms are:

1. `auto-correct`
2. `escalate-to-manager`
3. `escalate-to-HR`
4. `flag-for-audit`
5. `no-action`

High-confidence, low-risk anomalies can proceed to the Action Agent automatically after compliance approval. Lower-confidence detected anomalies enter the HITL queue with the safest proposed action, usually `flag-for-audit`. High-risk actions always require HITL even when confidence is high; confidence and blast radius are treated as separate safety axes.

## 7. Reinforcement Learning Feedback Layer

The RL decision is intentionally narrow: for a given anomaly context, rank the five action arms. This matches a contextual bandit better than PPO/REINFORCE because the assignment demo has only a few feedback cycles, while policy-gradient routing would need far more episodes to show a reliable change.

Algorithm: linear Thompson-sampling contextual bandit.

Context features are deterministic: anomaly type, confidence, severity, amount bucket, department hash, tenure bucket, probation flag, prior incident count, detector corroboration count, data-quality score, and cycle phase. The feature vector does not include free-form LLM text.

Reward sources:

| Source | Event | Reward |
|---|---|---|
| HITL | approve | `+1.0` |
| HITL | reject | `-1.0` |
| HITL | modify | partial reward based on edit distance from the proposed action |
| HITL | timeout | mild penalty, then safe default |
| Outcome | auto-correct recurs within N cycles | negative delayed reward |
| Outcome | auto-correct does not recur within N cycles | positive delayed reward |
| False positive | reviewer or eval labels anomaly as legitimate/data error | negative reward |
| Compliance | proposed action receives a hard veto | `-1.0` |

Rewards are summed per decision and clipped to keep a single event from destabilizing the policy. Every posterior update persists to disk: small bandit matrices in `data/rl_policy.npz`, plus auditable `rl_decisions`, `rl_rewards`, and `pending_rewards` tables in SQLite. Restarting the server reloads the learned policy; it does not reset behavior.

This is a real learning signal, not prompt stuffing. Human decisions and outcomes update numerical posteriors that change future action rankings. The Loom/demo compares action distribution before and after at least two simulated feedback cycles.

## 8. HITL, Compliance, and Memory

HITL is triggered when:

```text
needs_human = confidence < AUTO_ACTION_THRESHOLD
           or action.risk_tier >= RISK_THRESHOLD
           or compliance_result.requires_approval
```

The HITL packet contains anomaly evidence, proposed action, confidence score, agent reasoning, policy citations when relevant, RL ranking, compliance findings, and similar past incidents. The reviewer can approve, reject, or modify; rejection and modification reasons are required and become future memory. If the reviewer does not respond in time, the system resumes with `flag-for-audit` as the safe default.

Compliance vetoes are final. If the top-ranked action is vetoed, the veto is logged as a negative reward and the next non-vetoed ranked action is checked. `flag-for-audit` is the guaranteed safe fallback.

Resolved incidents are embedded into Chroma with context, action, outcome, reward, and metadata. On a similar future anomaly, memory applies weighted pseudo-observations to a cloned bandit posterior for that decision. This warm-starts ranking without permanently contaminating the global policy from one precedent. A confirmed similar incident also contributes a capped precedent term to `corroboration_boost`, so the second occurrence can score higher confidence without memory alone making a weak anomaly look certain. The demo shows the second occurrence of a similar anomaly receiving higher confidence, fewer graph hops, and either a faster HITL decision or no HITL when safe.

## 9. Data, RAG, and Mock APIs

Seeded generators create:

- 500-1,000 employees across departments, bands, countries, tenure buckets, managers, and probation states.
- Six months of payroll, attendance, leave, performance, and training records.
- Injected payroll, leave, and compliance anomalies with hidden ground-truth labels for calibration and evals.
- HR policies in Markdown/PDF covering leave, payroll corrections, overtime/attendance, and training/compliance.
- FastAPI mock services with OpenAPI schemas for payroll lookup/adjustment, attendance lookup, leave flagging, tickets, alerts, and audit logging.

Consistency checks make the artifacts defensible: payroll arithmetic balances, leave balances never go negative, attendance and leave do not overlap, and policy thresholds match both the YAML compliance rules and injected anomaly generator.

## 10. Observability, Evaluation, and Cost

Observability is structural, not optional. The `@node` wrapper around every graph node records step latency, tool calls, retries, token usage, cost, cache hits, selected RL action, sampled scores, reward events, and output summaries. JSONL traces support inspection per workflow run, while SQLite tables power a Streamlit live trace view.

The evaluation harness contains 15 declarative cases:

- Happy paths: grounded policy answer, reactive payslip investigation, scheduled scan across all anomaly classes, safe high-confidence action.
- Edge cases: small cohorts, mock API failures, HITL timeout, high confidence plus approval-required action.
- Adversarial cases: prompt injection, unsupported policy question, nonexistent employee.
- RL cases: manager rejects auto-corrections, approve-heavy reviewer, compliance-veto learning, memory warm-start on a repeat anomaly.

RL diagnostics render cumulative reward and action distribution shift across feedback cycles. These plots are the proof that the feedback layer changes proposals over time.

Cost tracking is centralized in one LLM client. Two modes run through the same eval harness:

- `naive`: all possible language work uses the strongest model tier, no prompt caching, no deterministic short-circuits.
- `optimized`: deterministic anomaly/compliance logic, small model for closed-set routing, mid-tier model for RAG synthesis, strongest model only for high-judgment narratives, prompt caching for static instructions/tool schemas, and narration only for user-visible or HITL cases.

The required >=20% reduction is reported from measured eval output, not estimated in prose.

## 11. Repository Shape

```text
src/selfheal/
  triggers/       # reactive, scheduled, alert intake, Signal model
  graph/          # state, supervisor, graph wiring, node tracing decorator
  agents/         # policy, anomaly, compliance, action, memory nodes
  anomaly/        # payroll, leave, compliance detectors and calibration
  rl/             # bandit, rewards, features, persistence, diagnostics
  hitl/           # approval API, interrupt/resume, timeout sweeper
  compliance/     # rules.yaml and rules engine
  rag/            # chunking, indexing, retrieval, verifier
  tools/          # mock tool clients and schemas
  mocks/          # FastAPI upstream services
  obs/            # traces, costs, structured logging
ui/               # Streamlit ops console and approvals
scripts/          # seed data, build index, run demo cycles, diagnostics
evals/            # 15-case harness and reviewer personas
data/             # SQLite, checkpoints, Chroma, traces, learned RL policy
```

## 12. Hardest Tradeoffs

**Bandit over REINFORCE:** The system learns action selection rather than long-horizon routing strategy. That is the right trade for the assignment because action choice has clear rewards, five discrete arms, and limited feedback cycles.

**Deterministic anomaly detection and compliance:** This sacrifices open-ended LLM discovery of novel anomaly types, but it gives auditability, calibration, repeatability, and most of the cost reduction.

**Embedded local stack:** SQLite, Chroma, APScheduler, and Streamlit minimize demo friction and make persistence easy to inspect. At production scale these become Postgres, a durable queue/workflow engine, a managed vector store, and a real reviewer console.

## 13. Production Path

For 3M employees and 750 clients, the design scales by replacing local infrastructure without changing the learning loop:

| Area | Demo | Production |
|---|---|---|
| Signals | In-process scheduler and FastAPI | Kafka/SQS or event bus with tenant partitioning |
| Workflow | LangGraph plus SQLite checkpointer | Durable workflow runtime such as LangGraph Platform or Temporal |
| Data | SQLite and pandas scans | Warehouse/Spark/Flink incremental detectors with drift monitors |
| RL | Global bandit policy | Hierarchical tenant policy: global prior plus tenant posterior, with off-policy evaluation |
| Memory | Embedded Chroma | Qdrant/pgvector with tenant namespaces, TTL, and PII controls |
| HITL | Streamlit and REST | SLA-aware reviewer queue, role-based approvals, four-eyes controls |
| Compliance | YAML rules | Versioned jurisdiction rule registry with legal review and rule tests |
| LLM ops | Static tier config | Gateway with budgets, rate limits, prompt release evals, and per-tenant cost controls |
