# Design Rationale

This document records the reasoning behind the contested or non-obvious design decisions in [architecture.md](architecture.md). The architecture document states *what* the system does; this document states *why*, the alternatives that were rejected, and the consequences the team is accepting.

It has two parts:

- **Part A — Decision Records.** Each deliberate design choice, in ADR form (Context / Decision / Consequences / Alternatives rejected). These are the items most likely to be challenged in design review.
- **Part B — Risk and Operability Register.** Operational concerns that are not single decisions — capacity, degradation, drift, and data lifecycle — documented as risks with mitigations and validation.

Status legend: **Accepted** — decided and reflected in the architecture. **Provisional** — accepted but expected to be revisited with production data.

---

# Part A — Decision Records

## DR-1: The RL layer ranks actions only; it has no execution authority

**Status:** Accepted

**Context.** A learning component in a payroll and compliance system is inherently risky. Thompson sampling explores by sampling from the posterior, so it will sometimes rank a non-greedy action first. We need the learning benefit without ever letting exploration cause an unauthorized side effect.

**Decision.** The bandit produces a *ranking* of the five action arms and nothing more. Three independent gates sit between the ranking and any side effect: the deterministic compliance veto, the risk-tier/confidence gate (high blast-radius or low-confidence routes to HITL), and human approval for everything in the HITL queue.

**Consequences.** Exploration manifests only as *which compliant action a human is asked to approve first*, or *which low-risk, compliance-cleared action auto-executes* — never as an unauthorized payroll change. The "cost" of exploration is a slightly suboptimal ordering of human review, which is cheap and self-correcting. The learning benefit is realized strictly within the safe set: among actions that are all allowed, the bandit learns which resolves a given context fastest and most correctly, reducing reviewer handling time and improving first-action precision on recurring families. The gates define the *feasible* set; the bandit ranks *quality* within it. Removing the bandit degrades efficiency, not safety.

**Alternatives rejected.** PPO/REINFORCE over Supervisor routing (rejected in architecture §12.3) requires exploration we cannot justify on payroll workflows. Letting the policy gate its own safety (no external gates) was rejected because it couples safety to model correctness.

## DR-2: Compliance vetoes feed a negative reward, even though the veto is deterministic

**Status:** Provisional

**Context.** A hard veto is already known from the rule registry and is enforced at runtime regardless of the bandit. Feeding it back as a `-1.0` reward risks teaching a stochastic learner something a rule already guarantees, which could be seen as injecting noise.

**Decision.** Emit a clipped `-1.0` veto reward, but treat it as a *ranking-efficiency* signal, not a safety signal.

**Consequences.** The reward discourages the bandit from spending its top rank on actions that will predictably be vetoed in a given context, which would otherwise push actionable work down the queue and inflate reviewer load. System correctness does not depend on this reward — the deterministic gate is the real enforcement — so it is safe to remove. We flag it Provisional: if production diagnostics show it adds more variance than ranking value, it is the first reward source we drop.

**Alternatives rejected.** Omitting the veto reward entirely (acceptable, but loses the queue-efficiency benefit). Using a larger penalty (rejected — would let one event dominate the clipped reward sum).

## DR-3: The "modify" reward is computed over action structure, not free text

**Status:** Accepted

**Context.** Partial reward for a reviewer's modification creates a reward-hacking risk: a reviewer who reflexively makes light tweaks could train the policy toward minimizing edit distance rather than toward correctness.

**Decision.** Edit distance is measured over the *action structure*, not narrative text. Changing action type (`auto-correct` → `escalate-to-HR`) is a large edit; adjusting a parameter within the same action type is a small edit. Reviewer modify-rate and edit-distance distributions are monitored per reviewer, and outlier reviewers' modifications are down-weighted. The independent delayed outcome reward dominates over time.

**Consequences.** A light tweak that preserves the action type yields near-full positive reward, which is the correct signal — the action choice was right. Persistent reviewer laziness is visible as a reviewer-quality metric and cannot durably steer the policy against real outcomes. Requires per-reviewer behavior monitoring as a first-class metric.

**Alternatives rejected.** Binary approve/reject only (rejected — discards the useful signal in a small correction). Free-text edit distance (rejected — trivially gameable and not meaningful).

## DR-4: Memory warm-starts a *cloned* posterior per decision, not the global policy

**Status:** Accepted

**Context.** Episodic memory should let repeated incident patterns be handled faster, but writing retrieved precedents directly into the global bandit would let a single precedent permanently contaminate the policy, and a bad precedent would compound.

**Decision.** Retrieved precedents are applied as *weighted pseudo-observations to a per-decision clone* of the global posterior, which is discarded after sampling. Only the real reward from the decision updates the global policy, once, through the normal path.

The mechanics, for review: a linear Thompson bandit maintains per arm a precision matrix `A = λI + Σ xᵢxᵢᵀ` and `b = Σ rᵢxᵢ`, with posterior mean `θ̂ = A⁻¹b` and covariance `σ²A⁻¹`. For one decision we clone to `(A', b')` and apply each precedent as `A' += w·xₚxₚᵀ`, `b' += w·rₚ·xₚ`, where `w ∈ (0,1]` is a capped similarity-and-recency weight. The action is sampled from `(A', b')`; the clone is then discarded.

**Consequences.** Three guarantees follow: (1) warm-start without permanent contamination — the precedent biases one decision only; (2) bad-precedent containment — capped weight, decays with dissimilarity, never compounds because it is re-derived per decision rather than accumulated; (3) no double-counting with confidence — the precedent's effect on *confidence* flows only through the capped `corroboration_boost` term (architecture §6), and its effect on *ranking* flows only through this pseudo-observation; separate posteriors, separate caps. Cost: a matrix clone and update per decision, negligible for a small action space.

**Alternatives rejected.** Direct global update from precedents (rejected — permanent contamination). Using memory only as reviewer-facing context with no ranking effect (rejected — forgoes the warm-start KPI).

## DR-5: Hierarchical policy — global prior with per-tenant posterior and partial pooling

**Status:** Accepted

**Context.** A new tenant has no learning history (cold start), while a very large tenant could swamp a naively pooled global prior that small tenants inherit.

**Decision.** Each tenant maintains its own posterior initialized from the global prior with inflated covariance. The global prior is a partial-pooling (shrinkage) estimate, not a raw sum: per-tenant contribution is capped and normalized roughly by `√n_tenant`. Global-prior updates are promoted only after off-policy evaluation.

**Consequences.** A new tenant explores more initially (high uncertainty) and converges as its own rewards arrive; it can be seeded from offline backtest on its historical incidents before live traffic. A million-employee tenant has more influence on the global prior than a tiny tenant, but not proportional to raw volume, so small tenants borrow strength without being overwritten. Cost: shrinkage weighting and an off-policy evaluation gate before any global-prior change reaches tenants.

**Alternatives rejected.** Single global policy (rejected — ignores tenant-specific patterns and lets large tenants dominate). Fully independent per-tenant policies (rejected — small tenants never escape cold start).

## DR-6: Delayed-reward attribution requires same-signature recurrence with no legitimate context change

**Status:** Accepted

**Context.** "Auto-correct recurs within N cycles → negative reward" is too blunt: an anomaly may recur because the employee's situation legitimately changed, not because the earlier action was wrong.

**Decision.** `N` is set per anomaly type to one correction cycle plus a grace window (payroll `N = 2` months; leave `N = 1` quarter). At horizon end the outcome checker re-runs the detector and attributes a negative delayed reward only when the *same anomaly signature* recurs for the *same employee* with *no intervening legitimate context change* (transfer, band change, comp revision, policy update). A legitimate change voids the attribution.

**Consequences.** The bandit is not penalized for correct actions taken in a world that later changed, and it does not learn to avoid correct actions whenever change is possible. Requires the outcome checker to detect intervening context changes, which it reads from HRIS lifecycle events.

**Alternatives rejected.** Penalize all recurrence (rejected — conflates relapse with legitimate change). No delayed reward (rejected — loses the strongest outcome-grounded signal).

## DR-7: Hard vetoes are final for automation; lawful exceptions are encoded as rules, not overrides

**Status:** Accepted

**Context.** Some jurisdictions have documented emergency exceptions (e.g. a payroll action permitted with VP sign-off). A literal "vetoes can never be bypassed" would make lawful exceptions impossible; an ad-hoc human override would create an unauditable path around compliance.

**Decision.** No human can override a hard veto. A lawful exception is modeled as a *versioned rule* whose precondition includes a recorded, authenticated approval artifact (e.g. VP sign-off). With the artifact present, the case is — by rule, deterministically — no longer a veto; without it, the veto stands.

**Consequences.** Every executable action remains traceable to a rule, never to an ad-hoc override, which is what an auditor requires. Adding an exception is a tested, reviewed rule change rather than a runtime escape hatch. The automated platform never executes a hard-vetoed action under any path.

**Alternatives rejected.** Reviewer-level veto override (rejected — unauditable, couples compliance to individual discretion). No exception mechanism at all (rejected — cannot represent real legal carve-outs).

## DR-8: The entailment verifier is an asymmetric filter, not a guarantee

**Status:** Accepted

**Context.** The Policy Agent's verifier is itself an LLM, so "who verifies the verifier" is a fair challenge. It cannot be treated as a correctness guarantee.

**Decision.** Treat the verifier as a probabilistic filter with measured error rates, made safe by asymmetry and defense in depth. A false *rejection* degrades to the safe "policy corpus does not cover this" answer. A false *acceptance* is the dangerous case, so the offline suite tracks the verifier's false-acceptance rate on an adversarial entailment fixture as the gating metric. Policy answers ground human-facing explanations and anomaly context only; they never directly cause a side effect, which still requires deterministic detection and the compliance gate. Citations are required and clickable so reviewers can confirm entailment themselves.

**Consequences.** The verifier raises the floor on grounded answers; humans and deterministic gates set the ceiling. No actionable decision depends on verifier output alone. Requires maintaining an adversarial entailment fixture and tracking false-acceptance rate as a release gate.

**Alternatives rejected.** Trusting the verifier as authoritative (rejected — single LLM check on a safety-relevant path). No verifier (rejected — more ungrounded claims reach reviewers).

---

# Part B — Risk and Operability Register

These are operational properties of the system rather than single decisions. Each is stated as a risk, the mitigation in the design, and how it is validated in production.

## RO-1: Scan fan-out and the checkpoint bottleneck

**Risk.** At millions of employees, naive per-employee graph runs and per-node checkpoint writes could overwhelm the workflow store.

**Mitigation.** Detection runs as a batch SQL/Spark job *outside* the graph; the overwhelming majority of employees produce no anomaly and never enter a graph run. A graph run is created only per detected anomaly — at a realistic sub-1% anomaly rate, a 5M-employee tenant yields tens of thousands of runs per cycle, not millions. Detection is partitioned by tenant; graph runs are queued and rate-limited per tenant so a noisy tenant cannot starve others; LLM calls are minimized so cost scales with anomalies, not headcount. The likely bottleneck is Postgres checkpoint write throughput (≈8–12 writes per run); see RO-2.

**Validation.** Load test the full pipeline at projected peak anomaly volume before launch; monitor checkpoint write latency and per-tenant run queue depth as SLOs.

## RO-2: Postgres as the checkpointer at scale

**Risk.** Checkpointing after every node creates write amplification.

**Mitigation.** Checkpoint volume scales with *anomalies × nodes*, not headcount (RO-1). Checkpoints are partitioned by tenant and time; completed-run checkpoints are compacted and archived to the warehouse on a short retention so the hot table stays small — only active and recently completed runs live in the low-latency store. Postgres is retained because HITL pause/resume, worker failover, and the idempotency-key-in-same-transaction guarantee (architecture §5) all require transactional, queryable, durable state that in-memory and object stores cannot provide.

**Validation.** Track hot-table size, write latency, and archive lag; alert on retention/compaction falling behind.

## RO-3: The system degrades safely when the LLM is unavailable

**Risk.** If the LLM gateway is down or misbehaving, the system must not block or mis-decide on the safety-critical paths.

**Mitigation.** The LLM has no decision authority on the action path. Anomaly detection (deterministic), compliance (deterministic rule evaluation), and action ranking (bandit over deterministic features) all run with the gateway down. A gateway-level kill switch makes LLM nodes optional enrichment that fail open to "route to human with raw evidence." Disabling the gateway loses only reactive natural-language intake and human-readable narratives; scheduled and system-alert paths are unaffected.

**Validation.** Periodic game-day exercise with the gateway disabled, asserting that detection → ranking → compliance → HITL still completes.

## RO-4: Silent degradation from calibration and bandit drift

**Risk.** The most dangerous failure is silent: if label latency grows or reviewers rubber-stamp, the confidence calibration and bandit posterior keep updating on stale or biased signal and decision quality erodes with no hard error firing.

**Mitigation.** Rolling recalibration with Expected Calibration Error tracking (architecture §6); reward-hacking checks and per-arm posterior trend diagnostics; false-positive sampling; reviewer agreement monitored as a *separate* metric from the calibration target (which uses independent ground truth — audit outcome, recurrence, blind second review). Critically, drift degrades efficiency and precision only — the deterministic compliance and HITL gates hold the safety line regardless of drift.

**Validation.** ECE and per-arm posterior dashboards with alert thresholds; policy promotion blocked on calibration and drift regression.

## RO-5: Tenant isolation

**Risk.** A retrieval or query returning another tenant's data would be a severe breach.

**Mitigation.** Namespace plus row-level security per store, enforced through a shared access layer rather than per-call discipline. A cross-tenant retrieval test runs in CI and as an online canary; any result carrying a foreign `tenant_id` fails the build and pages.

**Validation.** CI isolation test on every change; online canary assertion in production.

## RO-6: Subject data erasure (GDPR / data-subject requests)

**Risk.** A subject's data spans multiple stores with different legal retention bases, and the learning components complicate naive deletion.

**Mitigation.** A per-subject erasure sweep with store-specific handling:

| Store | Action on erasure | Rationale |
|---|---|---|
| Warehouse / operational tables | Delete or anonymize subject rows | No legal basis to retain after request |
| Incident-memory embeddings | Delete subject's incident vectors | Prevents the subject reappearing as retrieved precedent |
| Active checkpoints | Purge subject PII; complete or fail-safe an active run first | Avoid orphaned PII in paused state |
| Reward ledger | Retain, pseudonymized (subject ID tombstoned) | Legal/audit basis to retain the decision record; the bandit learned from aggregate features, not identity |
| Audit ledger | Retain, pseudonymized | Immutable compliance record under legal basis |

The bandit is not retrained on erasure: the policy is a function of aggregate feature statistics, not of any single identifiable subject, so removing a subject's future retrievability (memory and warehouse) satisfies the request without unwinding learned aggregate behavior.

**Validation.** Erasure runbook with a verification step per store; legal review of the retention basis per jurisdiction.

---

# Summary of accepted tradeoffs

- The product floor is a deterministic rules engine plus a reviewer queue; the LLM and bandit are efficiency multipliers with no safety authority. Removing either degrades throughput and precision, not safety (DR-1, RO-3).
- Learning is contained: per-decision warm-start (DR-4), per-tenant posteriors with partial pooling (DR-5), and outcome attribution guarded against spurious recurrence (DR-6).
- Compliance is final for automation, with lawful exceptions expressed as tested rules rather than overrides (DR-7).
- The largest residual risk is silent drift, addressed by monitoring rather than by a hard gate, because the safety gates are independent of model quality (RO-4).
