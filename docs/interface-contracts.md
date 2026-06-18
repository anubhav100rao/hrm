# Interface Contracts and Example Payloads

This document expands the architecture brief with concrete Python interfaces and example payloads. The goal is to make the handoffs between graph nodes, RAG, episodic memory, RL, HITL, compliance, and tools unambiguous.

All examples use Pydantic-style models. Exact implementation can split these classes across modules, but the contracts should remain stable.

## 1. Shared Enums and IDs

```python
from __future__ import annotations

from datetime import datetime, date
from typing import Annotated, Any, Literal, TypedDict
from pydantic import BaseModel, Field


TriggerSource = Literal["reactive", "scheduled", "system"]
IntentKind = Literal["policy_question", "action_request", "anomaly_report", "mixed"]
AnomalyType = Literal["payroll", "leave", "compliance"]
ActionType = Literal[
    "auto-correct",
    "escalate-to-manager",
    "escalate-to-HR",
    "flag-for-audit",
    "no-action",
]
DecisionKind = Literal["approve", "reject", "modify", "timeout"]
Coverage = Literal["full", "partial", "none"]
ComplianceEffect = Literal["allow", "require_approval", "veto"]
RewardSource = Literal["hitl", "outcome", "false_positive", "compliance"]
```

IDs are generated as ULIDs so they sort by creation time and stay readable in traces:

```text
sig_01JZ4Q0K8XRQ5EZEJTK7GRP2WX
run_01JZ4Q0N9EH9A7B5HMB8S4R3YK
anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW
dec_01JZ4Q12T31NQZNE9EFYE24C6A
```

## 2. Mock HR Data Models

These are the rows produced by the seeded generator and stored in SQLite. The detectors read these directly.

```python
class EmployeeRecord(BaseModel):
    employee_id: str
    name: str
    country: Literal["AE", "IN", "SG"]
    department: str
    band: Literal["B1", "B2", "B3", "B4", "B5"]
    manager_id: str | None
    hire_date: date
    on_probation: bool
    base_salary: float
    currency: str


class PayrollLine(BaseModel):
    employee_id: str
    period: str  # "2026-05"
    base_pay: float
    housing_allowance: float
    transport_allowance: float
    overtime_pay: float
    deductions: float
    net_pay: float
    currency: str
    run_id: str


class AttendanceSummary(BaseModel):
    employee_id: str
    period: str
    working_days: int
    leave_days: float
    unplanned_absence_days: float
    overtime_hours: float
    weekly_overtime_hours_max: float


class LeaveTransaction(BaseModel):
    leave_id: str
    employee_id: str
    leave_type: Literal["annual", "sick", "unpaid", "maternity", "bereavement"]
    start_date: date
    end_date: date
    days: float
    status: Literal["approved", "pending", "rejected"]
    adjacent_to_weekend: bool
    adjacent_to_holiday: bool


class TrainingRecord(BaseModel):
    employee_id: str
    course_id: str
    course_name: str
    due_date: date
    completed_at: date | None
    mandatory: bool
```

Example generated rows:

```json
{
  "employee": {
    "employee_id": "E0412",
    "name": "Aisha Khan",
    "country": "AE",
    "department": "Engineering",
    "band": "B3",
    "manager_id": "E0021",
    "hire_date": "2024-08-12",
    "on_probation": false,
    "base_salary": 18500.0,
    "currency": "AED"
  },
  "payroll_current": {
    "employee_id": "E0412",
    "period": "2026-05",
    "base_pay": 18500.0,
    "housing_allowance": 5500.0,
    "transport_allowance": 1200.0,
    "overtime_pay": 0.0,
    "deductions": 4200.0,
    "net_pay": 21000.0,
    "currency": "AED",
    "run_id": "payrun_2026_05_v1"
  },
  "payroll_previous": {
    "employee_id": "E0412",
    "period": "2026-04",
    "base_pay": 18500.0,
    "housing_allowance": 5500.0,
    "transport_allowance": 1200.0,
    "overtime_pay": 0.0,
    "deductions": 3400.0,
    "net_pay": 21800.0,
    "currency": "AED",
    "run_id": "payrun_2026_04_v1"
  },
  "ground_truth_eval_only": {
    "label": "payroll_outlier",
    "injected_reason": "unexpected deduction spike of AED 800",
    "visible_to_agents": false
  }
}
```

## 3. Signal and Intent Contracts

All triggers normalize into `Signal`. This keeps downstream nodes independent of whether the workflow started as chat, cron, or webhook.

```python
class Signal(BaseModel):
    signal_id: str
    source: TriggerSource
    received_at: datetime
    raw_payload: dict[str, Any]
    employee_ids: list[str] = Field(default_factory=list)
    priority: Literal["low", "normal", "high"] = "normal"
    thread_id: str | None = None


class Intent(BaseModel):
    kind: IntentKind
    user_text: str | None = None
    employee_ids: list[str] = Field(default_factory=list)
    period: str | None = None
    policy_area: str | None = None
    requested_action: ActionType | None = None
    confidence: float = Field(ge=0, le=1)
```

Example reactive signal:

```json
{
  "signal_id": "sig_01JZ4Q0K8XRQ5EZEJTK7GRP2WX",
  "source": "reactive",
  "received_at": "2026-06-18T10:15:22Z",
  "raw_payload": {
    "message": "Why was my payslip short by AED 800 this month?",
    "employee_id": "E0412"
  },
  "employee_ids": ["E0412"],
  "priority": "normal",
  "thread_id": "chat_E0412_20260618"
}
```

Example system alert signal:

```json
{
  "signal_id": "sig_01JZ4Q37JQ03BEBWCN2G6N9HTW",
  "source": "system",
  "received_at": "2026-06-18T10:18:00Z",
  "raw_payload": {
    "alert_type": "PAYRUN_VARIANCE",
    "employee_id": "E0412",
    "period": "2026-05",
    "delta_amount": -800.0,
    "currency": "AED"
  },
  "employee_ids": ["E0412"],
  "priority": "high"
}
```

## 4. RAG Interfaces

Policy chunks are stored in Chroma. The LLM sees only retrieved chunks, not the whole policy book.

```python
class PolicyChunk(BaseModel):
    chunk_id: str
    policy_name: str
    section_path: list[str]
    text: str
    country_scope: list[str]
    effective_from: date
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQuery(BaseModel):
    query: str
    employee_country: str | None = None
    policy_area: str | None = None
    top_k: int = 6


class RetrievedChunk(BaseModel):
    chunk: PolicyChunk
    score: float
    rank: int


class PolicyAnswer(BaseModel):
    answer: str
    citations: list[str]
    coverage: Coverage
    retrieved_chunks: list[RetrievedChunk]
    verifier_passed: bool
```

Example stored policy chunk:

```json
{
  "chunk_id": "leave_policy__ae__sick_leave__certificate_threshold",
  "policy_name": "Leave Policy",
  "section_path": ["Sick Leave", "Medical Certificate Requirement"],
  "text": "For AE employees, sick leave for more than 2 consecutive working days requires a medical certificate uploaded within 3 working days.",
  "country_scope": ["AE"],
  "effective_from": "2026-01-01",
  "metadata": {
    "policy_area": "leave",
    "source_file": "data/policies/leave_policy.md"
  }
}
```

Example RAG answer:

```json
{
  "answer": "For AE employees, sick leave longer than 2 consecutive working days requires a medical certificate within 3 working days. The current policy corpus does not mention a separate Engineering-specific exception.",
  "citations": ["leave_policy__ae__sick_leave__certificate_threshold"],
  "coverage": "partial",
  "retrieved_chunks": [
    {
      "score": 0.83,
      "rank": 1,
      "chunk": {
        "chunk_id": "leave_policy__ae__sick_leave__certificate_threshold",
        "policy_name": "Leave Policy",
        "section_path": ["Sick Leave", "Medical Certificate Requirement"],
        "text": "For AE employees, sick leave for more than 2 consecutive working days requires a medical certificate uploaded within 3 working days.",
        "country_scope": ["AE"],
        "effective_from": "2026-01-01",
        "metadata": {"policy_area": "leave"}
      }
    }
  ],
  "verifier_passed": true
}
```

## 5. Anomaly and Action Interfaces

The Anomaly Agent produces auditable evidence. RL ranks action candidates, but does not execute them.

```python
class EvidenceItem(BaseModel):
    name: str
    value: float | str | bool
    baseline: float | str | None = None
    explanation: str


class ActionCandidate(BaseModel):
    action_type: ActionType
    params: dict[str, Any] = Field(default_factory=dict)
    risk_tier: Literal["low", "medium", "high"]
    rationale: str


class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: AnomalyType
    employee_id: str
    period: str
    evidence: list[EvidenceItem]
    detector_scores: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    severity: float = Field(ge=0, le=1)
    recommended_action: ActionCandidate
    action_candidates: list[ActionCandidate]


class ProposedAction(BaseModel):
    decision_id: str
    anomaly_id: str
    action_type: ActionType
    params: dict[str, Any]
    risk_tier: Literal["low", "medium", "high"]
    source: Literal["prior", "rl", "hitl_modified", "safe_default"]
    rationale: str
```

Example anomaly:

```json
{
  "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
  "anomaly_type": "payroll",
  "employee_id": "E0412",
  "period": "2026-05",
  "evidence": [
    {
      "name": "month_over_month_net_pay_delta",
      "value": -800.0,
      "baseline": 0.0,
      "explanation": "Net pay dropped by AED 800 vs previous month without a matching approved deduction event."
    },
    {
      "name": "peer_cohort_robust_z",
      "value": -3.4,
      "baseline": "Engineering/B3/AE",
      "explanation": "The payroll delta is 3.4 MAD-scaled deviations from the peer cohort median."
    }
  ],
  "detector_scores": {
    "payroll_delta": 0.91,
    "deduction_mismatch": 0.88,
    "data_quality": 0.96
  },
  "confidence": 0.89,
  "severity": 0.42,
  "recommended_action": {
    "action_type": "auto-correct",
    "params": {
      "employee_id": "E0412",
      "period": "2026-05",
      "delta": 800.0,
      "currency": "AED"
    },
    "risk_tier": "medium",
    "rationale": "High confidence payroll deduction mismatch below the approval tier."
  },
  "action_candidates": [
    {
      "action_type": "auto-correct",
      "params": {"employee_id": "E0412", "period": "2026-05", "delta": 800.0, "currency": "AED"},
      "risk_tier": "medium",
      "rationale": "Reverse unexpected deduction if compliance allows."
    },
    {
      "action_type": "escalate-to-HR",
      "params": {"employee_id": "E0412", "period": "2026-05"},
      "risk_tier": "low",
      "rationale": "Ask HR payroll specialist to validate the deduction."
    },
    {
      "action_type": "flag-for-audit",
      "params": {"employee_id": "E0412", "period": "2026-05"},
      "risk_tier": "low",
      "rationale": "Keep an audit trail without changing payroll."
    }
  ]
}
```

## 6. RL Interfaces

The bandit ranks the same action candidates generated by the deterministic prior. It persists both the chosen action and all scores for diagnostics.

```python
class BanditContext(BaseModel):
    anomaly_id: str
    employee_id: str
    feature_names: list[str]
    x: list[float]
    memory_neighbors_used: list[str] = Field(default_factory=list)


class ArmScore(BaseModel):
    action_type: ActionType
    sampled_score: float
    posterior_mean: float
    rank: int


class BanditDecision(BaseModel):
    decision_id: str
    anomaly_id: str
    context: BanditContext
    scores: list[ArmScore]
    selected_action: ProposedAction
    policy_version: str
    rng_seed: int | None = None


class RewardEvent(BaseModel):
    reward_id: str
    decision_id: str
    source: RewardSource
    value: float
    reason: str
    observed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Example bandit decision after memory warm-start:

```json
{
  "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
  "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
  "context": {
    "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
    "employee_id": "E0412",
    "feature_names": [
      "type_payroll",
      "confidence",
      "severity",
      "amount_bucket_500_1000",
      "country_AE",
      "prior_incident_count",
      "memory_confirmed_similar"
    ],
    "x": [1.0, 0.89, 0.42, 1.0, 1.0, 1.0, 1.0],
    "memory_neighbors_used": ["inc_01JYZZ1G5S6D8XG8TKKJ3HHE2K"]
  },
  "scores": [
    {"action_type": "escalate-to-HR", "sampled_score": 0.71, "posterior_mean": 0.64, "rank": 1},
    {"action_type": "auto-correct", "sampled_score": 0.44, "posterior_mean": 0.38, "rank": 2},
    {"action_type": "flag-for-audit", "sampled_score": 0.22, "posterior_mean": 0.21, "rank": 3},
    {"action_type": "escalate-to-manager", "sampled_score": 0.13, "posterior_mean": 0.10, "rank": 4},
    {"action_type": "no-action", "sampled_score": -0.29, "posterior_mean": -0.25, "rank": 5}
  ],
  "selected_action": {
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
    "action_type": "escalate-to-HR",
    "params": {"employee_id": "E0412", "period": "2026-05", "reason": "Payroll deduction mismatch AED 800"},
    "risk_tier": "low",
    "source": "rl",
    "rationale": "Prior feedback rejected auto-corrections for similar payroll deductions; HR escalation has higher posterior reward."
  },
  "policy_version": "bandit_v3",
  "rng_seed": 42
}
```

Example reward events:

```json
[
  {
    "reward_id": "rew_01JZ4Q6R9X7V87PJ8A7BEMFMFV",
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "source": "hitl",
    "value": 1.0,
    "reason": "HR reviewer approved escalation",
    "observed_at": "2026-06-18T10:22:51Z",
    "metadata": {"reviewer_id": "hr_ops_17"}
  },
  {
    "reward_id": "rew_01JZ4R1ZP26AXH4E3ZQVYQ37MZ",
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "source": "outcome",
    "value": 0.3,
    "reason": "No recurrence detected within 3 cycles",
    "observed_at": "2026-06-18T11:05:00Z",
    "metadata": {"cycles_observed": 3}
  }
]
```

## 7. Compliance Interfaces

Compliance returns all matched rules, not just the first failure, so the trace explains the full decision.

```python
class RuleHit(BaseModel):
    rule_id: str
    name: str
    effect: ComplianceEffect
    severity: Literal["soft", "hard"]
    message: str


class ComplianceResult(BaseModel):
    proposed_action: ProposedAction
    effect: ComplianceEffect
    requires_approval: bool
    vetoed: bool
    hits: list[RuleHit]
    safe_fallback: ActionCandidate
```

Example compliance approval requirement:

```json
{
  "proposed_action": {
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
    "action_type": "auto-correct",
    "params": {"employee_id": "E0412", "period": "2026-05", "delta": 800.0, "currency": "AED"},
    "risk_tier": "medium",
    "source": "rl",
    "rationale": "High confidence payroll mismatch."
  },
  "effect": "require_approval",
  "requires_approval": true,
  "vetoed": false,
  "hits": [
    {
      "rule_id": "PAY-003",
      "name": "payroll_correction_tier",
      "effect": "require_approval",
      "severity": "soft",
      "message": "Payroll corrections above AED 500 require HR manager approval."
    }
  ],
  "safe_fallback": {
    "action_type": "flag-for-audit",
    "params": {"employee_id": "E0412", "period": "2026-05"},
    "risk_tier": "low",
    "rationale": "Safe fallback when correction cannot execute automatically."
  }
}
```

Example hard veto:

```json
{
  "effect": "veto",
  "requires_approval": false,
  "vetoed": true,
  "hits": [
    {
      "rule_id": "OT-001",
      "name": "weekly_overtime_cap",
      "effect": "veto",
      "severity": "hard",
      "message": "Action would legitimize overtime above the weekly cap."
    }
  ],
  "safe_fallback": {
    "action_type": "flag-for-audit",
    "params": {"employee_id": "E0198", "period": "2026-05"},
    "risk_tier": "low",
    "rationale": "Compliance veto requires manual audit."
  }
}
```

## 8. HITL and Resolution Interfaces

The approval packet is the exact object mirrored to SQLite and displayed in Streamlit.

```python
class ReviewPacket(BaseModel):
    approval_id: str
    run_id: str
    decision_id: str
    anomaly: Anomaly
    proposed_action: ProposedAction
    confidence: float
    agent_reasoning: str
    policy_citations: list[str]
    rl_ranking: list[ArmScore]
    compliance_result: ComplianceResult
    episodic_precedents: list["IncidentMatch"]
    expires_at: datetime


class HITLDecision(BaseModel):
    approval_id: str
    decision: DecisionKind
    reviewer_id: str | None = None
    reason: str | None = None
    modified_action: ProposedAction | None = None
    decided_at: datetime


class IncidentResolution(BaseModel):
    incident_id: str
    run_id: str
    anomaly: Anomaly
    final_action: ProposedAction
    hitl_decision: HITLDecision | None
    compliance_result: ComplianceResult
    tool_results: list["ToolResult"]
    reward_total: float
    outcome: Literal["resolved", "rejected", "vetoed", "expired", "tool_failed", "no_action"]
    resolution_summary: str
    resolved_at: datetime
```

Example review packet:

```json
{
  "approval_id": "appr_01JZ4Q3SCBZT9JK6K38RYVCPY7",
  "run_id": "run_01JZ4Q0N9EH9A7B5HMB8S4R3YK",
  "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
  "confidence": 0.89,
  "agent_reasoning": "Payroll evidence shows an unexpected AED 800 deduction. Prior similar incidents were approved for HR escalation rather than direct correction.",
  "policy_citations": ["payroll_policy__ae__correction_tiers"],
  "rl_ranking": [
    {"action_type": "escalate-to-HR", "sampled_score": 0.71, "posterior_mean": 0.64, "rank": 1},
    {"action_type": "auto-correct", "sampled_score": 0.44, "posterior_mean": 0.38, "rank": 2}
  ],
  "expires_at": "2026-06-18T10:45:00Z"
}
```

Example reviewer modification:

```json
{
  "approval_id": "appr_01JZ4Q3SCBZT9JK6K38RYVCPY7",
  "decision": "modify",
  "reviewer_id": "hr_ops_17",
  "reason": "Escalate to HR payroll queue first; do not adjust payroll until the deduction code owner confirms.",
  "modified_action": {
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
    "action_type": "escalate-to-HR",
    "params": {
      "employee_id": "E0412",
      "period": "2026-05",
      "queue": "payroll-corrections",
      "reason": "Unexpected AED 800 deduction"
    },
    "risk_tier": "low",
    "source": "hitl_modified",
    "rationale": "Reviewer requested verification before correction."
  },
  "decided_at": "2026-06-18T10:22:51Z"
}
```

Example final resolution stored for memory:

```json
{
  "incident_id": "inc_01JZ4Q8YBFGW9DCK7DPE2FS31N",
  "run_id": "run_01JZ4Q0N9EH9A7B5HMB8S4R3YK",
  "final_action": {
    "decision_id": "dec_01JZ4Q12T31NQZNE9EFYE24C6A",
    "anomaly_id": "anom_01JZ4Q0XB1AHZCSJEB69JWJ8NW",
    "action_type": "escalate-to-HR",
    "params": {
      "employee_id": "E0412",
      "period": "2026-05",
      "queue": "payroll-corrections",
      "reason": "Unexpected AED 800 deduction"
    },
    "risk_tier": "low",
    "source": "hitl_modified",
    "rationale": "Reviewer requested verification before correction."
  },
  "reward_total": 0.4,
  "outcome": "resolved",
  "resolution_summary": "HR escalation created ticket TCK-12991. Reviewer modified original auto-correct proposal because the deduction code required owner verification.",
  "resolved_at": "2026-06-18T10:23:04Z"
}
```

## 9. Episodic Memory Interfaces

The memory document is canonicalized before embedding. Metadata stays queryable.

```python
class IncidentMemoryRecord(BaseModel):
    incident_id: str
    embedded_text: str
    anomaly_type: AnomalyType
    employee_country: str
    employee_band: str
    action_taken: ActionType
    decision_source: Literal["auto", "hitl", "timeout", "compliance_fallback"]
    reward_total: float
    outcome: str
    cycle: int
    run_id: str


class IncidentMatch(BaseModel):
    incident_id: str
    similarity: float
    action_taken: ActionType
    reward_total: float
    outcome: str
    summary: str
```

Example Chroma memory document:

```json
{
  "id": "inc_01JZ4Q8YBFGW9DCK7DPE2FS31N",
  "document": "payroll anomaly | AE | B3 | net pay dropped 500-1000 AED | no matching approved deduction | action escalate-to-HR | reviewer modified from auto-correct | outcome resolved",
  "metadata": {
    "anomaly_type": "payroll",
    "employee_country": "AE",
    "employee_band": "B3",
    "action_taken": "escalate-to-HR",
    "decision_source": "hitl",
    "reward_total": 0.4,
    "outcome": "resolved",
    "cycle": 2,
    "run_id": "run_01JZ4Q0N9EH9A7B5HMB8S4R3YK"
  }
}
```

## 10. Tool Interfaces

Tools are side-effect boundaries. They are called only by the Action Agent after compliance and HITL gates.

```python
class ToolCall(BaseModel):
    tool_name: str
    idempotency_key: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    idempotency_key: str
    ok: bool
    status_code: int | None = None
    response: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float
    retries: int = 0
```

Example `POST /tickets` tool call:

```json
{
  "tool_name": "tickets.create",
  "idempotency_key": "run_01JZ4Q0N9EH9A7B5HMB8S4R3YK:escalate-to-HR:E0412",
  "args": {
    "queue": "payroll-corrections",
    "employee_id": "E0412",
    "period": "2026-05",
    "summary": "Unexpected AED 800 deduction",
    "evidence": [
      "net_pay_delta=-800",
      "peer_cohort_robust_z=-3.4",
      "no approved deduction event"
    ]
  }
}
```

Example result:

```json
{
  "tool_name": "tickets.create",
  "idempotency_key": "run_01JZ4Q0N9EH9A7B5HMB8S4R3YK:escalate-to-HR:E0412",
  "ok": true,
  "status_code": 201,
  "response": {
    "ticket_id": "TCK-12991",
    "queue": "payroll-corrections",
    "status": "open"
  },
  "error": null,
  "latency_ms": 146.3,
  "retries": 0
}
```

## 11. Graph State and Node Interfaces

Each node receives `GraphState` and returns a partial update. The graph runtime merges updates through reducers.

```python
class StateTransition(BaseModel):
    node: str
    entered_at: datetime
    exited_at: datetime
    input_digest: str
    output_summary: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class GraphState(TypedDict, total=False):
    run_id: str
    signal: Signal
    cycle: int
    intent: Intent
    policy_answer: PolicyAnswer
    anomalies: list[Anomaly]
    episodic_neighbors: list[IncidentMatch]
    bandit_decision: BanditDecision
    proposed_action: ProposedAction
    compliance_result: ComplianceResult
    review_packet: ReviewPacket
    hitl_decision: HITLDecision
    tool_results: list[ToolResult]
    incident_resolution: IncidentResolution
    rewards: list[RewardEvent]
    final_response: str
    state_log: list[StateTransition]
```

Node signatures:

```python
async def supervisor_node(state: GraphState) -> dict[str, Any]: ...

async def policy_agent_node(state: GraphState) -> dict[str, Any]: ...

async def anomaly_agent_node(state: GraphState) -> dict[str, Any]: ...

async def memory_recall_node(state: GraphState) -> dict[str, Any]: ...

async def action_ranker_node(state: GraphState) -> dict[str, Any]: ...

async def compliance_agent_node(state: GraphState) -> dict[str, Any]: ...

async def hitl_gate_node(state: GraphState) -> dict[str, Any]: ...

async def action_agent_node(state: GraphState) -> dict[str, Any]: ...

async def memory_write_node(state: GraphState) -> dict[str, Any]: ...

async def reward_update_node(state: GraphState) -> dict[str, Any]: ...
```

## 12. Interface Handoff Map

| Producer | Writes | Consumer | Purpose |
|---|---|---|---|
| Trigger layer | `Signal` | Supervisor | Normalize all trigger classes |
| Supervisor | `Intent` | Policy/Anomaly/Action request nodes | Decide graph path without direct agent calls |
| Policy Agent | `PolicyAnswer` | Anomaly, HITL, final response | Ground decisions and answers in cited policy |
| Anomaly Agent | `Anomaly` | Memory Recall, RL Ranker, HITL | Provide evidence, confidence, and candidates |
| Memory Recall | `IncidentMatch[]` | RL Ranker, HITL | Bias action ranking and show precedent |
| RL Ranker | `BanditDecision`, `ProposedAction` | Compliance Agent | Rank allowed actions using learned policy |
| Compliance Agent | `ComplianceResult` | HITL or Action Agent | Enforce hard veto and approval requirements |
| HITL Gate | `HITLDecision` | Action Agent, Reward Update | Capture human feedback |
| Action Agent | `ToolResult[]` | Memory Write, Reward Update | Execute side effects and record outcome |
| Memory Write | `IncidentMemoryRecord` | Future Memory Recall | Store resolved precedent |
| Reward Update | `RewardEvent[]` | RL policy store | Persist learning signal |

