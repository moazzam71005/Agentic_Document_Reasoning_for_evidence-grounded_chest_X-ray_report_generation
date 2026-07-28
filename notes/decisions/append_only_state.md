# Append-Only State

This note details the design of the agentic memory.

## What
The `EvidenceState` ($S_t$) maintains a history of all evidence items and retrieved guidelines in an append-only log across the iterations of the agent controller loop.

## Why
Allowing the agentic loop to overwrite or update state variables in place destroys historical context. Keeping the state append-only enables cross-view consistency checking (e.g. comparing global pathology predictions with local zone findings) and lets the planner refer back to previously requested views.

## Alternatives Considered & Outcome
* **In-Place State Mutability**: Modifying the current state variables with the latest view prediction. This prevents retrospective reasoning and consistency checking.
* **Outcome**: Implemented an append-only `EvidenceState` log which accumulates dictionaries of evidence and guideline packages over the agentic sequence.

---
*Related: [[decisions-moc]]*
