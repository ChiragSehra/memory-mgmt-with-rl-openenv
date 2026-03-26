from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import ConversationTurn, Episode, MemoryType


_PREFERENCES: List[Tuple[str, str]] = [
    ("ClickHouse", "clickhouse"),
    ("PostgreSQL", "postgresql"),
    ("SQLite", "sqlite"),
    ("Python", "python"),
    ("FastAPI", "fastapi"),
]

_CONSTRAINTS: List[Tuple[str, str]] = [
    ("Keep answers concise.", "concise"),
    ("Use UTC timestamps.", "utc"),
    ("Avoid external dependencies.", "dependencies"),
    ("Return bullet points.", "bullets"),
    ("Explain the tradeoffs.", "tradeoffs"),
]

_PROJECT_FACTS: List[Tuple[str, str]] = [
    ("The project goal is a memory agent.", "memory agent"),
    ("The evaluation should be deterministic.", "deterministic"),
    ("We need a fixed memory budget.", "budget"),
    ("The baseline should be rule based.", "rule based"),
]

_DISTRACTORS = [
    "By the way, I also like coffee.",
    "Let's talk about something unrelated.",
    "I walked my dog this morning.",
    "Can you ignore this random detail?",
]

_FINAL_QUERY_TEMPLATES = [
    "Given my previous preferences and constraints, answer the task in a way that matches them.",
    "Use what I told you earlier and respond to the request with the right format.",
    "Based on our earlier discussion, provide the final answer that respects my preferences.",
]


@dataclass
class SyntheticEpisodeGenerator:
    memory_budget: int = 200
    min_turns: int = 6
    max_turns: int = 8

    def generate(self, seed: int | None = None) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"episode_{episode_seed}_{rng.randint(1000, 9999)}"

        preference_text, preference_keyword = rng.choice(_PREFERENCES)
        constraint_text, constraint_keyword = rng.choice(_CONSTRAINTS)
        project_text, project_keyword = rng.choice(_PROJECT_FACTS)
        distractor_text = rng.choice(_DISTRACTORS)
        final_query = rng.choice(_FINAL_QUERY_TEMPLATES)

        turns: List[ConversationTurn] = []
        turns.append(
            ConversationTurn(
                turn_id=0,
                text=f"My preference is {preference_text}.",
                kind="preference",
                memory_type=MemoryType.PREFERENCE,
                metadata={"keyword": preference_keyword},
            )
        )
        turns.append(
            ConversationTurn(
                turn_id=1,
                text=distractor_text,
                kind="distractor",
            )
        )
        turns.append(
            ConversationTurn(
                turn_id=2,
                text=constraint_text,
                kind="constraint",
                memory_type=MemoryType.CONSTRAINT,
                metadata={"keyword": constraint_keyword},
            )
        )

        if rng.random() > 0.5:
            corrected_preference_text, corrected_preference_keyword = rng.choice(_PREFERENCES)
            turns.append(
                ConversationTurn(
                    turn_id=3,
                    text=f"Correction: actually use {corrected_preference_text} instead.",
                    kind="correction",
                    memory_type=MemoryType.PREFERENCE,
                    metadata={
                        "keyword": corrected_preference_keyword,
                        "correction_of": "preference",
                    },
                )
            )
            latest_preference_keyword = corrected_preference_keyword
            latest_preference_text = corrected_preference_text
        else:
            turns.append(
                ConversationTurn(
                    turn_id=3,
                    text=project_text,
                    kind="project_info",
                    memory_type=MemoryType.PROJECT_INFO,
                    metadata={"keyword": project_keyword},
                )
            )
            latest_preference_keyword = preference_keyword
            latest_preference_text = preference_text

        turns.append(
            ConversationTurn(
                turn_id=4,
                text=rng.choice(_DISTRACTORS),
                kind="distractor",
            )
        )
        turns.append(
            ConversationTurn(
                turn_id=5,
                text=f"{final_query} The answer should reflect {latest_preference_text} and {constraint_text.lower()}",
                kind="final_query",
            )
        )

        if rng.random() > 0.5 and len(turns) < self.max_turns:
            turns.insert(
                4,
                ConversationTurn(
                    turn_id=4,
                    text=rng.choice(_DISTRACTORS),
                    kind="distractor",
                ),
            )

        turns = [turn for turn in turns[: self.max_turns]]
        for index, turn in enumerate(turns):
            turns[index] = ConversationTurn(
                turn_id=index,
                text=turn.text,
                kind=turn.kind,
                memory_type=turn.memory_type,
                tags=turn.tags,
                metadata=turn.metadata,
            )

        required_memory_types = [MemoryType.PREFERENCE.value, MemoryType.CONSTRAINT.value]
        required_keywords = [latest_preference_keyword, constraint_keyword]
        episode_metadata: Dict[str, object] = {
            "required_memory_types": required_memory_types,
            "required_keywords": required_keywords,
            "final_query": turns[-1].text,
            "turn_count": len(turns),
            "latest_preference_keyword": latest_preference_keyword,
            "latest_constraint_keyword": constraint_keyword,
            "latest_project_keyword": project_keyword if any(turn.kind == "project_info" for turn in turns) else "",
        }

        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.memory_budget,
            metadata=episode_metadata,
        )
