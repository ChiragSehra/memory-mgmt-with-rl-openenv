# CLAUDE.md

## Project Overview

**Memory Management Agent with RL + OpenEnv** — a reinforcement learning system that trains an LLM-based policy to manage memory optimally during multi-turn conversations.

The core question: *Given a stream of user interactions, how should an agent allocate limited memory resources to maximize future usefulness?*

## Repository Structure

```
src/memory_management_agent/    # Main package
├── schemas.py          # All dataclasses and enums (MemoryItem, Action, Observation, etc.)
├── environment.py      # MemoryManagementEnv (OpenEnv-compatible)
├── episode.py          # SyntheticEpisodeGenerator
├── memory_store.py     # MemoryStore (budget-constrained, LRU eviction)
├── agents.py           # 6 baseline agents
├── grader.py           # Grader + RewardComposer
├── evaluation.py       # run_episode(), evaluate_split(), BenchmarkReport
├── training.py         # Prompt building, rollout collection, TRL scaffold
├── analysis.py         # Failure analysis, memory evolution tracking
├── review.py           # Report rendering
└── utils.py            # Shared utilities

tests/
└── test_core.py        # Full test suite (unittest)

memory-management-agent.md               # Design document
memory-management-agent-execution-plan.md  # Phase-by-phase implementation plan
```

## Key Architecture Concepts

### Data Flow

```
SyntheticEpisodeGenerator
    → MemoryManagementEnv (reset/step loop)
        → Agent (act) → Action
        → MemoryStore (add/query/update/delete)
        → Grader (score_episode) → GraderMetrics
        → RewardComposer (compose) → scalar reward
    → TRL Trainer (GRPO/PPO)
```

### Action Space (7 types)

| Action | Description |
|--------|-------------|
| `STORE(text)` | Save raw memory item |
| `STORE_SUMMARY(text)` | Save compressed version |
| `IGNORE` | Skip this turn |
| `RETRIEVE(ids)` | Fetch specific memories |
| `UPDATE(id, text)` | Modify existing memory |
| `DELETE(id)` | Remove memory |
| `ANSWER(text)` | Final response (terminal action) |

### Memory Types

- `PREFERENCE` — user preferences (e.g., "prefers ClickHouse")
- `CONSTRAINT` — hard constraints (e.g., "keep answers under 5 lines")
- `PROJECT_INFO` — facts about current project context

### Reward Formula

```
R = 0.45 * success
  + 0.20 * precision
  + 0.15 * recall
  + 0.10 * compactness
  + 0.10 * freshness
  - penalties
```

Dense (step-level) rewards fire immediately; delayed rewards fire at episode end.

## Development Commands

### Run Tests

```bash
python -m unittest tests/test_core.py -v
```

### Quick Smoke Test

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent, run_episode
)
env = MemoryManagementEnv(memory_budget=200)
agent = RuleBasedMemoryAgent()
result = run_episode(agent, env, seed=42)
print(result.reward)
```

### Run Baseline Experiment

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent,
    run_training_experiment, TrainingConfig
)
env = MemoryManagementEnv(memory_budget=200)
report = run_training_experiment(
    RuleBasedMemoryAgent(), env,
    train_seeds=tuple(range(10)),
    visible_eval_seeds=tuple(range(10, 15)),
    hidden_eval_seeds=tuple(range(5000, 5005)),
    output_dir="./checkpoints",
    config=TrainingConfig(run_name="baseline-test")
)
```

### Collect and Analyze Rollouts

```python
from src.memory_management_agent import collect_rollouts, analyze_rollouts, render_full_review
rollouts = collect_rollouts(agent, env, seeds=tuple(range(5)))
analysis = analyze_rollouts(rollouts)
render_full_review(analysis)
```

## Important Schemas

All schemas are in `schemas.py`. Key types:

- **`Observation`** — what the agent sees each step: `current_message`, `conversation_window`, `memory_bank`, `budget_remaining`, `step_number`
- **`Action`** — agent output: `type: ActionType`, optional `text`, optional `ids`
- **`MemoryItem`** — stored memory: `id`, `text`, `type`, `created_at`, `last_used`, `token_length`, `utility_score`
- **`GraderMetrics`** — 14 scoring dimensions including `success`, `precision`, `recall`, `compactness`, `freshness`, `non_interference`
- **`EpisodeResult`** — final outcome: `episode`, `answer`, `metrics`, `reward`, `trace`

## Baseline Agents (for comparison)

| Agent | Strategy |
|-------|----------|
| `NoMemoryAgent` | Always IGNORE |
| `StoreEverythingAgent` | Always STORE |
| `PreferenceOnlyAgent` | Store only PREFERENCE/CONSTRAINT/PROJECT_INFO |
| `KeywordRetrievalAgent` | Keyword-based retrieval heuristics |
| `EmbeddingRetrievalAgent` | Semantic embedding retrieval |
| `RuleBasedMemoryAgent` | Most sophisticated rule-based baseline |

The RL-trained policy must outperform all baselines on hidden eval seeds.

## Evaluation Design

- **Visible seeds** (1–4999): training + validation
- **Hidden seeds** (5000+): holdout, used only for final evaluation
- `generalization_gap = visible_reward - hidden_reward` — detects overfitting/reward hacking
- `evaluate_split()` returns a `BenchmarkReport` with both splits

## Environment Configuration

```python
MemoryManagementEnv(
    memory_budget=200,   # Token budget for memory store
    max_turns=8          # Max turns per episode
)
```

## Known Failure Modes

| Mode | Description | Penalty |
|------|-------------|---------|
| Memory hoarding | Stores everything, wastes budget | `memory_bloat_penalty` |
| Over-retrieval | Retrieves irrelevant memories | `-0.12` per irrelevant retrieval |
| Under-retrieval | Fails to retrieve key info at answer time | `-0.5` delayed |
| Stale memory | Stores correction but keeps old fact | `contradiction_penalty` |
| Reward hacking | Learns shortcuts that don't generalize | Caught by hidden eval |

## Phase Status

- ✅ Phase 0–5: Schemas, environment, memory store, grading, baselines, evaluation, training scaffold, analysis
- 🔄 Phase 6: TRL training loop connection (scaffold exists in `training.py`, needs model integration)
- Planned: Vector DB backend, memory decay, LLM judge grading, production deployment

## Issue Tracking

This project uses **beads (bd)** for issue tracking. See `AGENTS.md` for workflow.
