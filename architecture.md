# Self-Healing HR Ops Platform - Production Architecture

## 1. Scope

This document describes a production-grade Darwinbox HR operations intelligence layer. The platform reacts to natural-language requests, runs scheduled and streaming anomaly scans, consumes upstream system alerts, executes guarded corrective actions, and continuously improves action recommendations from human feedback and operational outcomes.

The design assumes multi-tenant enterprise scale: millions of employees, hundreds of customers, multiple jurisdictions, strict auditability, tenant data isolation, high availability, and explicit controls around payroll and compliance side effects. The Python services use typed interfaces and durable infrastructure rather than in-memory or notebook-style orchestration.

Companion contract docs:

- [interface-contracts.md](interface-contracts.md) defines the Pydantic interfaces and example payloads for signals, RAG chunks, anomalies, RL decisions, HITL packets, incidents, resolutions, rewards, tools, and graph state.
- [llm-instructions.md](llm-instructions.md) defines the LLM instruction contracts and structured outputs for Supervisor triage, RAG synthesis, verification, reviewer narrative, and final user response.

## 2. Capability Mapping

| Capability | Design choice |
|---|---|
| Supervisor plus at least 4 sub-agents | LangGraph `StateGraph` with Supervisor, Policy Agent, Anomaly Detection Agent, Compliance Agent, Action Agent, plus Memory nodes |
| No direct agent-to-agent calls | Agents only read and write shared `GraphState`; conditional graph edges route all transitions |
| Three trigger classes | Reactive REST/chat requests, scheduled workflows, and event/webhook alerts all normalize into one `Signal` envelope |
| RAG grounded in HR policies | Managed vector index, header-aware chunking, cited answers, and "not covered" fallback for missing policy support |
| Anomaly detection with confidence and action | Deterministic statistics/rules produce calibrated confidence and a 5-action candidate set |
| RL feedback layer | Contextual bandit ranks the 5 actions and updates from HITL, outcomes, false positives, and compliance vetoes |
| HITL approval | LangGraph `interrupt()` pauses high-risk or low-confidence decisions; approval API and reviewer console resume runs |
| Compliance rules | Versioned rule registry evaluated by a deterministic rules service; hard vetoes cannot be overridden |
| Episodic memory | Managed vector store persists resolved incidents and warm-starts action ranking for similar future cases |
| Observability and evaluation | Structured traces, offline/online evaluation, RL diagnostics, token/cost ledger, and production SLOs |

## 3. High-Level Flow

At the highest level, the system is a stateful workflow engine wrapped around HR-specific agents. Every input, whether it is a chat request, scheduled scan, or upstream alert, becomes the same internal object: a `Signal`. From that point onward, the source no longer matters; the graph treats every workflow as a sequence of state transitions over the same `GraphState`.

```text
                                  +-----------------------------+
                                  | Enterprise data sources     |
                                  | HRIS/payroll/attendance     |
                                  | policy docs + rule registry |
                                  | audit labels + outcomes     |
                                  +--------------+--------------+
                                                 |
                                                 v
+----------------------+  +----------------------+  +----------------------+
| Reactive request     |  | Scheduled scan       |  | System alert         |
| Web/mobile/API       |  | Temporal/Airflow     |  | Kafka/webhook/event  |
| API Gateway/FastAPI  |  | batch + streaming    |  | upstream platforms   |
+----------+-----------+  +----------+-----------+  +----------+-----------+
           |                         |                         |
           +------------ normalize into typed Signal -----------+
                                     |
                                     v
                            +--------+---------+
                            | LangGraph        |
                            | StateGraph       |
                            | Postgres checkpoint|
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
        | managed vector  | | SQL/Spark/Python| | validation      |
        | embedding svc   | | calibrated conf | | tool params     |
        +--------+--------+ +--------+--------+ +--------+--------+
                 |                   |                   |
                 +-------------------+-------------------+
                                     |
                                     v
                            +--------+---------+
                            | Memory Recall    |
                            | Qdrant/pgvector  |
                            | incident memory  |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | RL Action Ranker |
                            | Thompson bandit  |
                            | feature store    |
                            | policy store     |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Compliance Agent |
                            | rule registry    |
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
              | Reviewer API/UI  |        | idempotency      |
              +--------+---------+        +--------+---------+
                       |                           |
                       +-------------+-------------+
                                     |
                                     v
                            +--------+---------+
                            | Memory Write     |
                            | incident record  |
                            | reward metadata  |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Reward Update    |
                            | bandit posterior |
                            | policy store     |
                            | reward ledger    |
                            +--------+---------+
                                     |
                                     v
                            +--------+---------+
                            | Observability    |
                            | OpenTelemetry    |
                            | metrics/logs     |
                            | cost ledger      |
                            +--------+---------+
                                     |
                                     v
                         Final response / tool result / queue item
```

### 3.1 End-to-End Lifecycle

1. **Inputs enter through three trigger paths.** Reactive requests come from web/mobile/API channels, scheduled scans run through durable workflow schedulers, and system-generated alerts arrive through events or webhooks from upstream platforms. All three paths normalize into a `Signal` containing source, payload, tenant ID, priority, employee IDs when known, and trace identifiers.

2. **The graph owns orchestration.** LangGraph receives the `Signal` and loads or creates the matching `GraphState`. A Postgres-backed checkpointer persists state after every node, which is what makes multi-turn conversation, HITL pause/resume, worker failover, and restart recovery possible.

3. **The Supervisor routes, but does not solve.** It performs lightweight triage: scheduled and system alerts go to anomaly investigation; reactive requests are classified into policy question, action request, anomaly report, or mixed. The Supervisor writes routing intent to state, and LangGraph conditional edges invoke the next node. No sub-agent calls another sub-agent directly.

4. **Policy grounding and anomaly detection enrich the same state.** The Policy Agent retrieves policy chunks and writes cited answers or policy context. The Anomaly Agent writes detected anomalies, evidence, severity, confidence, and an initial recommended action. If the request is mixed, both outputs can coexist in state before action selection.

5. **Episodic memory adds precedent before action ranking.** The Memory Recall node retrieves similar resolved incidents from a managed vector store. These incidents are shown to humans and used to warm-start the bandit for this decision, so repeated patterns can be handled faster and with more confidence.

6. **The RL layer ranks actions.** The contextual bandit receives deterministic features from the anomaly, employee, policy context, and memory signals. It ranks the five allowed actions: `auto-correct`, `escalate-to-manager`, `escalate-to-HR`, `flag-for-audit`, and `no-action`. The top-ranked action becomes the proposed action, not an executed action.

7. **Compliance is the hard safety boundary.** The Compliance Agent evaluates the proposed action against YAML rules. A hard veto blocks execution, emits a negative reward to the bandit, and causes the graph to try the next safe ranked action. Human approval cannot override a hard veto.

8. **HITL handles uncertainty and risk.** If confidence is below the auto-action threshold, risk tier is high, or compliance requires approval, the graph pauses with `interrupt()`. The approval UI/API shows evidence, confidence, policy citations, reasoning, RL ranking, compliance findings, and precedents. The reviewer can approve, reject, or modify. Timeout resumes with `flag-for-audit`.

9. **The Action Agent executes only after approval and compliance.** Tool calls go through typed integration adapters with schema validation, idempotency keys, retries for transient errors, circuit breakers, and fallback to audit when execution is unsafe or fails semantically.

10. **Learning closes the loop.** HITL decisions, delayed outcome checks, false-positive labels, and compliance vetoes produce reward events. The bandit posterior is updated and persisted to disk, so later runs measurably change their action rankings.

11. **Resolved incidents become future memory.** The Memory Write node stores the final incident context, action, human decision, outcome, and reward in the managed incident-memory index. Similar future incidents retrieve this precedent.

12. **Observability is emitted throughout, not bolted on.** Every node appends a `StateTransition` to shared state: node name, input digest, output summary, latency, tool calls, token usage, cost, selected RL action, and rewards. The same event is exported through OpenTelemetry and indexed for reviewer UI, incident forensics, evaluation, and cost reporting.

### 3.2 What Persists Where

| Artifact | Store | Why it exists |
|---|---|---|
| Workflow state and HITL checkpoints | Postgres-backed LangGraph checkpointer | Resume paused runs, support worker failover, and preserve audit history |
| Employee/payroll/leave/training data | Operational databases, warehouse, and feature store | Queryable HR source of truth and detector inputs |
| Policy chunks and incident memory | Managed vector store | RAG retrieval and episodic warm-start |
| RL policy | Policy store plus reward ledger | Persist learned behavior and audit reward history |
| Trace and cost events | OpenTelemetry, log store, metrics store, cost warehouse | Reviewer UI, SLOs, evaluation, cost reporting, debugging |
| Compliance rules | Versioned rule registry | Auditable hard-veto logic outside prompts |

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
| `IncidentResolution` | Memory Write | Managed episodic memory | Store resolved precedent for future warm-start |
| `RewardEvent` | HITL/outcome/compliance hooks | RL policy store | Update learned action policy |

Full Python definitions and JSON examples are in [interface-contracts.md](interface-contracts.md).

## 5. Agents

**Policy Agent:** Retrieves from the HR policy corpus using a managed vector store. Policies are split by markdown headings and clauses so numeric limits stay with their conditions. The agent returns cited answers only; if retrieved chunks do not support an answer, it says the policy corpus does not cover the question. A verifier checks that claims are entailed by cited chunks.

**Anomaly Detection Agent:** Uses deterministic pandas/numpy detectors, not LLM judgment. Payroll outliers use robust z-scores within peer cohorts such as department, band, and country. Leave abuse uses clustering around weekends/holidays, policy-limit pressure, and threshold-gaming patterns. Compliance violations include missing training, overtime cap breaches, probation constraints, and correction-window issues.

**Compliance Agent:** Evaluates every proposed action against YAML rules before execution. Rules include overtime caps, leave notice periods, probation restrictions, payroll correction tiers, mandatory training gates, retro-correction windows, and double-correction locks. A hard veto blocks the action even if RL, the Supervisor, or a human reviewer preferred it.

**Action Agent:** Executes only compliance-approved actions through production integration adapters: payroll adjustment, leave flag, ticket escalation, audit log, and training enrollment. Guardrails include JSON-schema validation, idempotency keys, parameter re-checks, retries for transient failures, circuit breakers, and sandbox mode for pre-production validation.

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

Confidence is calibrated against historical incident labels, payroll audit outcomes, reviewer decisions, and backtesting datasets:

```text
confidence = clamp01(calibrated_signal_strength * data_quality * corroboration_boost)
```

`calibrated_signal_strength` and `data_quality` are each in `[0, 1]`. `corroboration_boost` is a bounded multiplier in `[1.0, 1.0 + MAX_BOOST]`, and the final product is clamped to `[0, 1]` so it always honors the `confidence: Field(ge=0, le=1)` contract. Signal strength comes from detector-specific statistics, such as robust z-score magnitude or tail probability. Data quality penalizes small cohorts, missing fields, and legitimate context changes. Corroboration increases confidence when independent signals agree, for example payroll delta plus unmatched overtime data, but the cap prevents corroboration alone from making a weak anomaly look certain.

The five action arms are:

1. `auto-correct`
2. `escalate-to-manager`
3. `escalate-to-HR`
4. `flag-for-audit`
5. `no-action`

High-confidence, low-risk anomalies can proceed to the Action Agent automatically after compliance approval. Lower-confidence detected anomalies enter the HITL queue with the safest proposed action, usually `flag-for-audit`. High-risk actions always require HITL even when confidence is high; confidence and blast radius are treated as separate safety axes.

## 7. Reinforcement Learning Feedback Layer

The RL decision is intentionally narrow: for a given anomaly context, rank the five action arms. This matches a contextual bandit better than PPO/REINFORCE because production feedback is sparse, delayed, and safety-constrained; policy-gradient routing would need far more exploration than payroll/compliance workflows can safely tolerate.

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
| False positive | reviewer or audit labels anomaly as legitimate/data error | negative reward |
| Compliance | proposed action receives a hard veto | `-1.0` |

Rewards are summed per decision and clipped to keep a single event from destabilizing the policy. Every posterior update is persisted to the policy store and mirrored into an auditable reward ledger. Worker restarts reload the learned policy; they do not reset behavior.

This is a real learning signal, not prompt stuffing. Human decisions and outcomes update numerical posteriors that change future action rankings. Production dashboards compare action distribution, approval rate, veto rate, false-positive rate, and reward curves across time windows and tenants.

## 8. HITL, Compliance, and Memory

HITL is triggered when:

```text
needs_human = confidence < AUTO_ACTION_THRESHOLD
           or action.risk_tier >= RISK_THRESHOLD
           or compliance_result.requires_approval
```

The HITL packet contains anomaly evidence, proposed action, confidence score, agent reasoning, policy citations when relevant, RL ranking, compliance findings, and similar past incidents. The reviewer can approve, reject, or modify; rejection and modification reasons are required and become future memory. If the reviewer does not respond in time, the system resumes with `flag-for-audit` as the safe default.

Compliance vetoes are final. If the top-ranked action is vetoed, the veto is logged as a negative reward and the next non-vetoed ranked action is checked. `flag-for-audit` is the guaranteed safe fallback.

Resolved incidents are embedded into the incident memory collection with context, action, outcome, reward, and metadata. On a similar future anomaly, memory applies weighted pseudo-observations to a cloned bandit posterior for that decision. This warm-starts ranking without permanently contaminating the global policy from one precedent. A confirmed similar incident also contributes a capped precedent term to `corroboration_boost`, so repeat patterns can score higher confidence without memory alone making a weak anomaly look certain. The production KPI is reduced reviewer handling time and improved approval precision on recurring incident families.

## 9. Data, RAG, and Integration APIs

Production data inputs:

- HRIS employee master data across tenants, departments, bands, countries, managers, and lifecycle states.
- Payroll, attendance, leave, performance, and training records from operational stores and warehouse tables.
- Historical incidents, payroll audits, reviewer decisions, compliance vetoes, and outcome labels for calibration and evaluation.
- HR policies from versioned document repositories, legal policy systems, and jurisdiction-specific rule registries.
- Integration APIs with explicit OpenAPI/JSON schemas for payroll lookup/adjustment, attendance lookup, leave flagging, ticketing, alerts, and audit logging.

Data-quality checks run before signals reach the graph: payroll arithmetic must balance, leave balances cannot go negative, attendance and leave cannot overlap, policy thresholds must match rule-registry thresholds, tenant IDs must remain isolated, and PII fields must be redacted from prompts and traces unless explicitly needed.

## 10. Observability, Evaluation, and Cost

Observability is structural, not optional. The `@node` wrapper around every graph node records step latency, tool calls, retries, token usage, cost, cache hits, selected RL action, sampled scores, reward events, and output summaries. Trace events are exported through OpenTelemetry and correlated with tenant, run, workflow, and reviewer identifiers.

Evaluation has three layers:

- **Offline regression suite:** deterministic fixtures for policy grounding, anomaly scoring, compliance rules, tool-schema validation, prompt-injection resistance, and reward calculations.
- **Backtesting:** replay historical incidents and audits to measure precision, recall, calibration, action distribution, veto rate, and reviewer agreement before policy promotion.
- **Online monitoring:** canary releases, tenant-level drift detection, false-positive sampling, reward-hacking checks, and SLO dashboards for latency, approval backlog, and incident resolution time.

RL diagnostics render cumulative reward, action distribution shift, per-arm posterior trends, approval rate, veto rate, and false-positive rate across time windows. Policy promotion requires offline backtest thresholds and online guardrail checks.

Cost tracking is centralized in one LLM client. Production cost controls include:

- deterministic anomaly/compliance logic that avoids unnecessary LLM calls;
- small models for closed-set routing and verification;
- stronger models only for high-judgment narratives;
- prompt caching for static instructions and schemas;
- retrieval-bounded prompts;
- tenant-level budgets, alerts, and rate limits.

Cost dashboards report spend per tenant, workflow type, node, model tier, and policy version.

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
  tools/          # integration adapters and schemas
  integrations/   # payroll, attendance, leave, ticketing, audit clients
  obs/            # traces, costs, structured logging
ui/               # reviewer console and operations views
jobs/             # scheduled scans, replays, diagnostics, policy promotion
tests/            # unit, integration, backtest, safety regression suites
infra/            # deployment manifests, dashboards, alerts, runbooks
docs/             # source PDF only
README.md
architecture.md
interface-contracts.md
llm-instructions.md
```

## 12. Technology Tradeoffs and Rationale

The main design bias is: use auditable, horizontally scalable infrastructure for workflow state, tenant data, policy retrieval, human approvals, and learning. The platform should fail closed around payroll and compliance, expose every decision to audit, and allow model or policy changes only through evaluation gates.

### 12.1 Workflow Orchestration: LangGraph State Machine plus Durable Runtime

| Option | Pros | Cons |
|---|---|---|
| LangGraph `StateGraph` with Postgres checkpointer | Clear shared-state graph, conditional routing, interrupt/resume, replayable state transitions, and strong separation between agents | Requires careful versioning of state schemas and graph migrations |
| Temporal-only workflow | Excellent durability, retries, timers, and operational controls | Less natural for LLM/RAG state transitions and agent graph inspection |
| Hand-rolled state machine | Maximum control | Rebuilds checkpointing, replay, HITL pause/resume, schema evolution, and transition tracing |
| CrewAI/AutoGen-style orchestration | Quick multi-agent conversations | Harder to enforce strict shared-state boundaries and deterministic compliance gates |

**Decision:** use LangGraph for the agent state graph and back it with durable Postgres checkpoints. For long-running schedules, retries, and operational timers, wrap graph execution in a durable worker runtime such as Temporal or an equivalent internal workflow platform.

**Accepted tradeoff:** two layers of orchestration. LangGraph owns semantic state transitions; the durable runtime owns retries, timers, queues, and worker lifecycle.

### 12.2 LLM Integration: Governed Gateway and Thin Client

| Option | Pros | Cons |
|---|---|---|
| Internal LLM gateway plus thin service client | Centralized budgets, model routing, prompt versioning, PII controls, retries, usage accounting, and audit logs | Gateway becomes a critical dependency and needs SLOs |
| Direct provider SDK in every agent | Maximum feature access and low initial complexity | Scattered cost accounting, uneven retry behavior, harder policy enforcement |
| Heavy chain framework across all nodes | Many prebuilt abstractions | Can obscure token/cost attribution and make deterministic governance harder |

**Decision:** route all LLM calls through a governed gateway and expose a thin typed client to graph nodes. Machine-consumed outputs must use schemas; deterministic nodes avoid LLM calls entirely.

**Accepted tradeoff:** slower adoption of provider-specific features. The upside is consistent safety, cost attribution, prompt rollout, and tenant-level controls.

### 12.3 RL Algorithm: Contextual Bandit

| Option | Pros | Cons |
|---|---|---|
| Linear Thompson-sampling contextual bandit | Sample-efficient, interpretable, easy to persist, and directly learns which of the 5 actions to rank first | Only learns a one-step action decision, not full long-horizon workflow strategy |
| LinUCB | Also sample-efficient and interpretable | Requires explicit exploration tuning; less natural for memory-based pseudo-observations |
| PPO/REINFORCE on Supervisor routing | More general sequential optimization | Requires exploration that is difficult to justify in payroll/compliance workflows; harder to audit |
| Learned reward model | Can capture nuanced human preferences | Adds model-risk, drift monitoring, and explainability burden |
| Prompt-only feedback memory | Simple to ship | Does not provide a measurable or auditable learning signal |

**Decision:** use Thompson sampling over deterministic context features. The action space is small, rewards are available from human decisions and outcomes, and posterior movement is easy to audit.

**Accepted tradeoff:** it optimizes action recommendations, not the whole graph policy. That is intentional: routing should remain explainable and rule-bounded, while action ranking can safely learn under compliance and HITL gates.

### 12.4 Anomaly Detection: Deterministic Statistics Before LLMs

| Option | Pros | Cons |
|---|---|---|
| SQL/Spark/Python statistical detectors with calibration | Reproducible, auditable, cost-efficient, and easy to validate against historical outcomes | Detects defined anomaly families better than unknown unknowns |
| LLM-based anomaly scoring | Flexible and easy to explain in natural language | Non-deterministic, hard to calibrate, expensive for scans, and risky for payroll/compliance claims |
| Unsupervised ML models such as Isolation Forest | Can surface unfamiliar outliers | Harder to explain to HR reviewers; needs per-tenant calibration and drift controls |
| Fully supervised classifiers | Strong performance when labels are rich | Labels are sparse, tenant-specific, and delayed; retraining governance is heavier |

**Decision:** use deterministic detectors for payroll, leave, and compliance anomalies. The LLM can narrate evidence for humans, but it does not decide that a payroll anomaly exists.

**Accepted tradeoff:** less open-ended discovery. Unsupervised detectors can feed low-risk audit queues, but automatic or semi-automatic action stays grounded in explainable signals.

### 12.5 Persistence: Postgres, Warehouse, Object Store, and Policy Store

| Option | Pros | Cons |
|---|---|---|
| Postgres for workflow/checkpoints/approvals | Strong transactional semantics, mature operations, row-level tenancy controls | Needs careful partitioning and retention policy at large scale |
| Warehouse/lakehouse for analytical history | Efficient backtests, drift analysis, and long retention | Not suitable for low-latency workflow mutation |
| Object store for artifacts and model/policy snapshots | Durable, cheap, versioned storage | Requires metadata indexing elsewhere |
| In-memory state | Low latency | Not acceptable for HITL, audit, or learned policy persistence |

**Decision:** use Postgres for graph checkpoints, approvals, reward ledger, and operational metadata; warehouse tables for historical analysis; object storage for versioned policy snapshots and large artifacts.

**Accepted tradeoff:** more infrastructure than a single database. The separation keeps latency-sensitive workflow state distinct from analytical backtesting and long-term retention.

### 12.6 Vector Store and Embeddings: Managed Vector Search

| Option | Pros | Cons |
|---|---|---|
| Qdrant or managed vector DB | Strong filtering, tenant namespaces, operational scaling, and vector-specific indexing | Additional service to operate and secure |
| pgvector | Keeps vectors close to relational metadata and tenant controls | Can be harder to scale independently for large vector workloads |
| Search engine hybrid retrieval | Strong lexical plus semantic search | More complex ranking and index management |
| Local embedded vector store | Simple | Not appropriate for multi-tenant production scale |
| Hosted embeddings | Strong quality and operational simplicity | Cost, vendor dependency, and PII routing requirements |
| Self-hosted embeddings | Data residency and cost control | Requires model serving and quality monitoring |

**Decision:** use a managed vector store with tenant namespaces and metadata filters. Choose hosted or self-hosted embeddings based on data residency, latency, and cost constraints per deployment region.

**Accepted tradeoff:** more operational surface. The benefit is reliable retrieval, policy isolation, and incident-memory scale across tenants and jurisdictions.

### 12.7 API and Reviewer UI: FastAPI Services plus Product Console

| Option | Pros | Cons |
|---|---|---|
| FastAPI services | Typed contracts, async support, OpenAPI schemas, straightforward integration testing | Requires separate auth, rate-limit, and deployment patterns |
| Existing internal service framework | Consistent with platform standards | May slow iteration if framework is heavy |
| React/product console | Best reviewer workflow, permissions, accessibility, and audit UI | Requires frontend delivery and design investment |
| Admin-only lightweight UI | Faster to build | Not sufficient for high-volume reviewer operations |

**Decision:** expose graph ingress, approvals, tool callbacks, and diagnostics through FastAPI-compatible services with strict OpenAPI contracts. Build the reviewer experience into the product console with role-based access, queue management, and audit views.

**Accepted tradeoff:** more product work up front. HITL is the primary feedback loop, so reviewer ergonomics and permissions are production requirements, not optional polish.

### 12.8 Scheduling and Eventing: Durable Workflows and Event Bus

| Option | Pros | Cons |
|---|---|---|
| Temporal schedules/workflows | Durable timers, retries, visibility, and long-running workflow support | Operational dependency and workflow modeling overhead |
| Airflow/Dagster | Strong batch orchestration and data dependencies | Less ideal for per-incident pause/resume and HITL |
| Kafka/SQS event bus | Scalable event ingestion and tenant partitioning | Requires idempotency and consumer-lag operations |
| In-process scheduler | Simple | Not reliable for production failover or horizontal workers |

**Decision:** use an event bus for upstream alerts and durable workflow scheduling for scans, HITL timeouts, delayed outcome checks, and reward resolution.

**Accepted tradeoff:** operational complexity. This is required because missed scans, stuck approvals, or lost delayed rewards directly affect payroll and compliance outcomes.

### 12.9 Compliance Engine: Versioned Rule Registry

| Option | Pros | Cons |
|---|---|---|
| Versioned YAML/JSON decision tables with safe evaluator | Auditable, reviewable by compliance teams, diffable, testable, and deterministic | Less expressive than arbitrary code |
| Python rule functions | Maximum flexibility and testability | Rules become code, not configuration; harder for HR/legal stakeholders to review |
| LLM-as-compliance-judge | Easy to add natural-language nuance | Unsafe for hard vetoes; non-deterministic and hard to audit |
| Business rules platform | Mature governance and non-engineer workflows | Adds vendor/platform complexity and integration overhead |

**Decision:** use a versioned rule registry with deterministic evaluation over `{tenant, jurisdiction, employee, anomaly, action}`. All matching rules are logged; any hard veto blocks execution.

**Accepted tradeoff:** rules are intentionally constrained. This is a good thing for compliance: hard vetoes should be predictable, tested, approved, and boring.

### 12.10 Observability: OpenTelemetry, Metrics, Audit Ledger

| Option | Pros | Cons |
|---|---|---|
| OpenTelemetry traces plus metrics/logs | Standard correlation across services, tenants, tools, and LLM calls | Requires sampling, retention, and PII controls |
| Audit ledger | Immutable record for compliance, reviewer decisions, vetoes, and tool side effects | Adds storage and query design requirements |
| Console logs only | Simple | Insufficient for incident review, SLOs, cost governance, and audits |

**Decision:** every node emits structured trace events through OpenTelemetry, with immutable audit events for approvals, vetoes, rewards, and side effects.

**Accepted tradeoff:** more metadata discipline. The benefit is end-to-end traceability across graph state, human decisions, model calls, and downstream systems.

## 13. Production Operating Model

Production readiness depends on reliability, governance, and tenant isolation as much as model quality.

| Area | Production control |
|---|---|
| Signals | Event bus partitioned by tenant, idempotency keys, replay support, dead-letter queues |
| Workflow | Durable graph checkpoints, run versioning, pause/resume, timeout recovery, migration plan |
| Data | Tenant-isolated access, warehouse backtests, feature-store lineage, drift monitors |
| RL | Hierarchical policy: global prior plus tenant posterior, off-policy evaluation before promotion |
| Memory | Tenant namespaces, PII minimization, retention policies, deletion workflows |
| HITL | SLA-aware reviewer queues, role-based approvals, four-eyes controls, escalation paths |
| Compliance | Versioned jurisdiction rule registry, legal review, unit tests, release approvals |
| LLM ops | Gateway budgets, rate limits, prompt release gates, data residency controls |
| Security | RBAC/ABAC, encryption, secrets management, audit exports, incident response runbooks |
| Reliability | SLOs for scan latency, approval backlog, action execution, and incident resolution |
