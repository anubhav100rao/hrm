# Self-Healing HR Ops Platform - Architecture Brief

## 1. Scope

This design targets the Darwinbox advanced assignment: an agentic HR operations engine that can react to natural-language requests, run scheduled anomaly scans, consume upstream system alerts, execute guarded corrective actions, and learn from human feedback over time.

The implementation is a single Python 3.11+ repo with generated mock artifacts: 500-1,000 employees, 3-5 pages of HR policy documents, seeded payroll/attendance/leave/performance data, and a FastAPI mock API spec. The architecture is local-first for the demo, but each storage and orchestration choice has a clear production replacement.

Companion contract docs:

- `docs/interface-contracts.md` defines the Pydantic interfaces and example payloads for signals, RAG chunks, anomalies, RL decisions, HITL packets, incidents, resolutions, rewards, tools, and graph state.
- `docs/llm-instructions.md` defines the LLM instruction contracts and structured outputs for Supervisor triage, RAG synthesis, verification, reviewer narrative, and final user response.

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

At the highest level, the system is a stateful workflow engine wrapped around HR-specific agents. Every input, whether it is a chat request, scheduled scan, or upstream alert, becomes the same internal object: a `Signal`. From that point onward, the source no longer matters; the graph treats every workflow as a sequence of state transitions over the same `GraphState`.

```text
                                  +-----------------------------+
                                  | Generated artifacts         |
                                  | SQLite: HR data             |
                                  | Markdown/PDF: policies      |
                                  | YAML: compliance rules      |
                                  +--------------+--------------+
                                                 |
                                                 v
+----------------------+  +----------------------+  +----------------------+
| Reactive request     |  | Scheduled scan       |  | System alert         |
| Streamlit chat       |  | APScheduler          |  | FastAPI webhook      |
| FastAPI REST         |  | logical demo cycles  |  | mock upstream API    |
+----------+-----------+  +----------+-----------+  +----------+-----------+
           |                         |                         |
           +------------ normalize into typed Signal -----------+
                                     |
                                     v
                            +--------+---------+
                            | LangGraph        |
                            | StateGraph       |
                            | SQLite checkpoint|
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Supervisor       |
                            | structured LLM   |
                            | routing output   |
                            +--------+---------+
                                     |
                 +-------------------+-------------------+
                 |                   |                   |
                 v                   v                   v
        +--------+--------+ +--------+--------+ +--------+--------+
        | Policy Agent    | | Anomaly Agent   | | Action request  |
        | RAG             | | pandas/numpy    | | Pydantic schema |
        | Chroma policies | | robust scoring  | | validation      |
        | local embeddings| | calibrated conf | | tool params     |
        +--------+--------+ +--------+--------+ +--------+--------+
                 |                   |                   |
                 +-------------------+-------------------+
                                     |
                                     v
                            +--------+---------+
                            | Memory Recall    |
                            | Chroma incidents |
                            | local embeddings |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | RL Action Ranker |
                            | Thompson bandit  |
                            | numpy + npz      |
                            | SQLite rewards   |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Compliance Agent |
                            | YAML rules       |
                            | safe evaluator   |
                            | hard vetoes      |
                            +--------+---------+
                                     |
                       +-------------+-------------+
                       |                           |
                       v                           v
              +--------+---------+        +--------+---------+
              | HITL Gate        |        | Action Agent     |
              | LangGraph        |        | FastAPI tools    |
              | interrupt()      |        | JSON schemas     |
              | Streamlit/API UI |        | idempotency      |
              +--------+---------+        +--------+---------+
                       |                           |
                       +-------------+-------------+
                                     |
                                     v
                            +--------+---------+
                            | Memory Write     |
                            | Chroma incident  |
                            | reward metadata  |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Reward Update    |
                            | bandit posterior |
                            | rl_policy.npz    |
                            | SQLite audit     |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Observability    |
                            | JSONL traces     |
                            | SQLite metrics   |
                            | cost ledger      |
                            +--------+---------+
                                     |
                                     v
                         Final response / tool result / queue item
```

### 3.1 End-to-End Lifecycle

1. **Inputs enter through three trigger paths.** Reactive requests come from chat or REST, scheduled scans run over the generated dataset, and system-generated alerts arrive through the mock upstream API. All three paths normalize into a `Signal` containing source, payload, priority, employee IDs when known, and trace identifiers.

2. **The graph owns orchestration.** LangGraph receives the `Signal` and loads or creates the matching `GraphState`. The checkpointer persists the state after every node, which is what makes multi-turn conversation, HITL pause/resume, and server restart recovery possible.

3. **The Supervisor routes, but does not solve.** It performs lightweight triage: scheduled and system alerts go to anomaly investigation; reactive requests are classified into policy question, action request, anomaly report, or mixed. The Supervisor writes routing intent to state, and LangGraph conditional edges invoke the next node. No sub-agent calls another sub-agent directly.

4. **Policy grounding and anomaly detection enrich the same state.** The Policy Agent retrieves policy chunks and writes cited answers or policy context. The Anomaly Agent writes detected anomalies, evidence, severity, confidence, and an initial recommended action. If the request is mixed, both outputs can coexist in state before action selection.

5. **Episodic memory adds precedent before action ranking.** The Memory Recall node retrieves similar resolved incidents from Chroma. These incidents are shown to humans and used to warm-start the bandit for this decision, so repeated patterns can be handled faster and with more confidence.

6. **The RL layer ranks actions.** The contextual bandit receives deterministic features from the anomaly, employee, policy context, and memory signals. It ranks the five allowed actions: `auto-correct`, `escalate-to-manager`, `escalate-to-HR`, `flag-for-audit`, and `no-action`. The top-ranked action becomes the proposed action, not an executed action.

7. **Compliance is the hard safety boundary.** The Compliance Agent evaluates the proposed action against YAML rules. A hard veto blocks execution, emits a negative reward to the bandit, and causes the graph to try the next safe ranked action. Human approval cannot override a hard veto.

8. **HITL handles uncertainty and risk.** If confidence is below the auto-action threshold, risk tier is high, or compliance requires approval, the graph pauses with `interrupt()`. The approval UI/API shows evidence, confidence, policy citations, reasoning, RL ranking, compliance findings, and precedents. The reviewer can approve, reject, or modify. Timeout resumes with `flag-for-audit`.

9. **The Action Agent executes only after approval and compliance.** Tool calls go through typed mock APIs with schema validation, idempotency keys, retries for transient errors, and fallback to audit when execution is unsafe or fails semantically.

10. **Learning closes the loop.** HITL decisions, delayed outcome checks, false-positive labels, and compliance vetoes produce reward events. The bandit posterior is updated and persisted to disk, so later runs measurably change their action rankings.

11. **Resolved incidents become future memory.** The Memory Write node stores the final incident context, action, human decision, outcome, and reward in Chroma. Similar future incidents retrieve this precedent.

12. **Observability is emitted throughout, not bolted on.** Every node appends a `StateTransition` to shared state: node name, input digest, output summary, latency, tool calls, token usage, cost, selected RL action, and rewards. The same event is written to JSONL and indexed in SQLite for the live trace UI and evaluation harness.

### 3.2 What Persists Where

| Artifact | Store | Why it exists |
|---|---|---|
| Workflow state and HITL checkpoints | LangGraph SQLite checkpointer | Resume paused runs and survive restart |
| Employee/payroll/leave/training data | SQLite | Queryable simulated HR source of truth |
| Policy chunks and incident memory | Chroma | RAG retrieval and episodic warm-start |
| RL policy | `data/rl_policy.npz` plus SQLite reward tables | Persist learned behavior and audit reward history |
| Trace and cost events | JSONL plus SQLite indexes | Live UI, eval assertions, cost comparison, debugging |
| Compliance rules | YAML | Auditable hard-veto logic outside prompts |

### 3.3 Safety Invariants

- Sub-agents never call each other; all communication happens through `GraphState`.
- The LLM never directly executes tools; it can only produce structured proposals that pass validation, compliance, and risk gates.
- Compliance hard vetoes are final and feed negative reward back into RL.
- Low-confidence or high-risk actions pause for human review.
- Timeout never auto-corrects; it falls back to `flag-for-audit`.
- Learned RL policy ranks actions but cannot bypass compliance, HITL, tool schemas, or idempotency checks.

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

The contract surface is intentionally explicit:

| Interface | Producer | Consumer | Purpose |
|---|---|---|---|
| `Signal` | Trigger layer | Supervisor | Normalize reactive, scheduled, and system-generated inputs |
| `PolicyAnswer` | Policy Agent | HITL/final response/anomaly context | Ground answers and decisions in cited policy chunks |
| `Anomaly` | Anomaly Agent | Memory/RL/HITL | Carry evidence, confidence, severity, and candidate actions |
| `BanditDecision` | RL Action Ranker | Compliance/HITL/diagnostics | Store ranked actions and sampled/posterior scores |
| `ComplianceResult` | Compliance Agent | HITL/Action Agent/RL rewards | Enforce approval requirements and hard vetoes |
| `ReviewPacket` | HITL Gate | Reviewer UI/API | Present context for approve/reject/modify decisions |
| `IncidentResolution` | Memory Write | Chroma episodic memory | Store resolved precedent for future warm-start |
| `RewardEvent` | HITL/outcome/compliance hooks | RL policy store | Update learned action policy |

Full Python definitions and JSON examples are in `docs/interface-contracts.md`.

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
docs/             # assignment PDF, interface contracts, LLM instruction contracts
```

## 12. Technology Tradeoffs and Rationale

The main design bias is: use boring, inspectable infrastructure for the demo, and spend complexity only where the assignment explicitly evaluates it - stateful agent orchestration, grounded RAG, HITL, compliance vetoes, episodic memory, and a real RL feedback loop.

### 12.1 Agent Orchestration: LangGraph

| Option | Pros | Cons |
|---|---|---|
| LangGraph `StateGraph` | Native shared state, conditional routing, checkpointing, interrupt/resume support, and a clean way to prove "no direct agent-to-agent calls" | Extra dependency; API surface is more complex than a simple loop |
| Hand-rolled state machine | Maximum control, fewer dependencies, easy to debug for a tiny prototype | We would need to rebuild checkpointing, HITL pause/resume, transition logging, and graph replay - exactly the features being evaluated |
| CrewAI/AutoGen | Fast conversational agent demos | These frameworks naturally encourage agent-to-agent conversations, which conflicts with the assignment's hard state-graph requirement |

**Decision:** use LangGraph. It maps directly to the required architecture: Supervisor routes, sub-agents write to shared state, all transitions are logged, and HITL can pause without blocking a process thread.

**Accepted tradeoff:** some graph boilerplate. This is worth it because the boilerplate is visible proof of the architecture constraint rather than hidden framework magic.

### 12.2 LLM Integration: Thin Provider Client, Not Full LangChain

| Option | Pros | Cons |
|---|---|---|
| Raw provider SDK behind `llm/client.py` | Full control over structured outputs, token usage, cache policy, model tiering, and cost accounting | We write small wrappers for retries, parsing, and tracing |
| LangChain chains everywhere | Lots of ready-made abstractions for prompts, retrievers, and chains | Harder to keep exact token accounting and cache behavior; can obscure which call made which decision |
| Provider-agnostic abstraction from day one | Easier future vendor swap | Adds indirection before the product requirements are stable |

**Decision:** use a thin LLM client module. Machine-consumed outputs are structured with schemas, every call is costed centrally, and deterministic nodes avoid LLM calls entirely.

**Accepted tradeoff:** mild provider coupling. The code isolates that coupling in one module, which is enough for the assignment and keeps cost optimization measurable.

### 12.3 RL Algorithm: Contextual Bandit

| Option | Pros | Cons |
|---|---|---|
| Linear Thompson-sampling contextual bandit | Sample-efficient, interpretable, easy to persist, and directly learns which of the 5 actions to rank first | Only learns a one-step action decision, not full long-horizon workflow strategy |
| LinUCB | Also sample-efficient and interpretable | Requires exploration tuning; warm-starting from episodic memory is less natural |
| PPO/REINFORCE on Supervisor routing | More general sequential learning story | Needs far more episodes; demo changes after 2 cycles would likely be noisy or artificial |
| Prompt-only feedback memory | Very simple | Not a real learning signal and explicitly would not satisfy the assignment |

**Decision:** use Thompson sampling over deterministic context features. The action space is small, rewards are clear, and the Loom can show posterior movement plus action-distribution shift after two feedback cycles.

**Accepted tradeoff:** it optimizes action recommendations, not the whole graph policy. That is intentional: routing is mostly deterministic from trigger type, while action choice has meaningful human feedback.

### 12.4 Anomaly Detection: Deterministic Statistics Before LLMs

| Option | Pros | Cons |
|---|---|---|
| Pandas/numpy detectors with calibration | Reproducible, auditable, cheap, and easy to validate against seeded ground truth | Detects known anomaly classes rather than inventing new ones |
| LLM-based anomaly scoring | Flexible and easy to explain in natural language | Non-deterministic, hard to calibrate, expensive for scans, and risky for payroll/compliance claims |
| Unsupervised ML models such as Isolation Forest | Can find unfamiliar outliers | Harder to explain to HR reviewers; still needs calibration and per-cohort tuning |

**Decision:** use deterministic detectors for payroll, leave, and compliance anomalies. The LLM can narrate evidence for humans, but it does not decide that a payroll anomaly exists.

**Accepted tradeoff:** less open-ended discovery. For production, an unsupervised "suggestion only" detector could feed `flag-for-audit`, but automatic action should stay grounded in explainable signals.

### 12.5 Persistence: SQLite plus Files

| Option | Pros | Cons |
|---|---|---|
| SQLite, JSONL, and `rl_policy.npz` | Zero ops, inspectable, transactional enough for local demo, survives restarts, easy for reviewers to query | Not ideal for multi-process or high-concurrency production workloads |
| Postgres | Strong concurrency, operational familiarity, natural production fit | Adds setup friction for the assignment and makes the local demo heavier |
| In-memory state | Fastest to prototype | Fails the persistence requirement for RL and HITL |

**Decision:** use SQLite for application state, approval queues, reward audit trails, cost ledger, and LangGraph checkpoints; use `rl_policy.npz` for compact bandit matrices.

**Accepted tradeoff:** single-process bias. This is fine for the assignment because persistence and inspectability matter more than horizontal scale.

### 12.6 Vector Store and Embeddings: Chroma plus Local Embeddings

| Option | Pros | Cons |
|---|---|---|
| Chroma embedded store | Local, persistent, no service dependency, good enough for small policy and incident corpora | Weaker filtering and scaling story than dedicated vector databases |
| Qdrant | Better production vector database with stronger filtering and indexing | Requires running and configuring another service |
| pgvector | Keeps vectors and relational metadata together | Needs Postgres and more setup |
| Hosted embeddings | Strong quality on broad corpora | API cost, network dependency, and less deterministic local evaluation |
| Local sentence-transformer embeddings | Free, reproducible, offline, fast for small corpora | Lower quality than top hosted embedding models on long or messy text |

**Decision:** use Chroma with local embeddings for both policy RAG and episodic memory. The corpus is intentionally small: 3-5 pages of policy plus resolved incidents.

**Accepted tradeoff:** not the final production vector architecture. At scale, this becomes Qdrant or pgvector with tenant namespaces, PII controls, and lifecycle policies.

### 12.7 API and Reviewer UI: FastAPI plus Streamlit

| Option | Pros | Cons |
|---|---|---|
| FastAPI | Typed request/response models, async support, automatic OpenAPI spec, good mock-tool ergonomics | More structure than a tiny Flask app |
| Flask | Very small and familiar | Less schema-first by default; more manual OpenAPI work |
| Django | Batteries included | Too heavy for a prototype mock API and agent service |
| Streamlit | Fast Python-only reviewer UI, good for showing traces and HITL packets in a demo | Less polished and less customizable than a real React product UI |
| React | Production-grade UI flexibility | Slower to build; adds a separate frontend stack not central to the assignment |

**Decision:** FastAPI for mock upstream APIs, action tools, alert intake, and approval endpoints; Streamlit for the ops console and approval queue.

**Accepted tradeoff:** the UI is demo-grade. The APIs are still clean enough that a production React console can replace Streamlit later.

### 12.8 Scheduling and Feedback Cycles: APScheduler plus Logical Cycles

| Option | Pros | Cons |
|---|---|---|
| APScheduler | Simple in-process cron, no broker, easy demo setup | Not a distributed scheduler |
| Celery or RQ | Durable background jobs and retries | Requires broker/worker setup; heavier than needed |
| System cron | Simple operational model | Harder to coordinate with app state and demo cycle controls |
| Logical cycle counter | Deterministic evals and fast two-cycle RL demo | Less realistic than waiting on wall-clock time |

**Decision:** APScheduler handles real scheduled jobs and timeout sweeps, while the demo/eval uses a logical `cycle` integer for recurrence windows and feedback comparisons.

**Accepted tradeoff:** simulated time is less production-real, but it makes before/after RL behavior reproducible and easy to grade.

### 12.9 Compliance Engine: YAML Rules plus Safe Expression Evaluation

| Option | Pros | Cons |
|---|---|---|
| YAML/JSON rules with a small evaluator | Auditable, editable without prompt changes, versionable, and directly satisfies the assignment | Less expressive than arbitrary code |
| Python rule functions | Maximum flexibility and testability | Rules become code, not configuration; harder for HR/legal stakeholders to review |
| LLM-as-compliance-judge | Easy to add natural-language nuance | Unsafe for hard vetoes; non-deterministic and hard to audit |
| Business rules engine | Mature rule management | Adds operational complexity for a small ruleset |

**Decision:** use 10-15 YAML rules evaluated by a constrained expression engine over `{employee, anomaly, action}`. All matching rules are logged; any hard veto blocks execution.

**Accepted tradeoff:** rules are intentionally simple. This is a good thing for compliance: hard vetoes should be predictable, tested, and boring.

### 12.10 Observability: JSONL, SQLite Metrics, and Optional UI

| Option | Pros | Cons |
|---|---|---|
| JSONL traces plus SQLite indexes | Easy to inspect, diff, test, and show in a demo | Not a full distributed tracing stack |
| OpenTelemetry from day one | Production-grade trace correlation | More setup and viewer infrastructure than the assignment needs |
| Console logs only | Minimal effort | Hard to power eval assertions, cost reports, and live trace UI |

**Decision:** every node emits structured trace events to JSONL and SQLite. This supports the assignment's required trace fields, eval assertions, cost accounting, and live reasoning trace.

**Accepted tradeoff:** local observability only. In production, the same event shape can be exported through OpenTelemetry.

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
