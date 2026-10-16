"""Per-session conversation memory with a size budget.

Only the user request and the final answer of each turn are kept (not raw tool
output), and the oldest turns are dropped first once the budget is exceeded.
This keeps follow-up questions working without letting context grow unbounded.
"""

from __future__ import annotations

from collections import defaultdict


class ConversationMemory:
    def __init__(self, char_budget: int = 12_000, max_sessions: int = 1_000) -> None:
        self.char_budget = char_budget
        self.max_sessions = max_sessions
        self._turns: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def history(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        turns = self._turns.get(session_id, [])
        kept: list[tuple[str, str]] = []
        used = 0
        for user, assistant in reversed(turns):
            size = len(user) + len(assistant)
            if used + size > self.char_budget:
                break
            kept.append((user, assistant))
            used += size
        messages: list[dict[str, str]] = []
        for user, assistant in reversed(kept):
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def save(self, session_id: str | None, user: str, assistant: str) -> None:
        if not session_id or not assistant:
            return
        if session_id not in self._turns and len(self._turns) >= self.max_sessions:
            self._turns.pop(next(iter(self._turns)))
        self._turns[session_id].append((user, assistant))
