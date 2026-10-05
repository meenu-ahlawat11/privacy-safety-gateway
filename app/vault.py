"""In-memory vault mapping sensitive values to opaque tokens."""

import re

_TOKEN_RE = re.compile(r"<[A-Z_]+_\d+>")


class Vault:
    """Tokenizes sensitive values and restores them later.

    Two hash maps are used because both directions are needed:
    ``value_to_token`` keyed by ``(type, value)`` gives O(1) average
    reuse of an existing token, and ``token_to_value`` keyed by the
    token string gives O(1) average lookup during restore. A dict is
    the right structure here: the keys are unique, we only need
    exact-match lookups, and hash maps provide O(1) average
    insert/lookup (O(n) worst case). This vault lives in memory only:
    values are never written to disk and never logged.
    """

    def __init__(self) -> None:
        self._token_to_value: dict[str, tuple[str, str]] = {}
        self._value_to_token: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def tokenize(self, type: str, value: str) -> str:
        """Return the token for (type, value), creating one if needed."""
        existing = self._value_to_token.get((type, value))
        if existing is not None:
            return existing
        count = self._counters.get(type, 0) + 1
        self._counters[type] = count
        token = f"<{type}_{count}>"
        self._value_to_token[(type, value)] = token
        self._token_to_value[token] = (type, value)
        return token

    def restore(self, text: str) -> str:
        """Replace known tokens in text with their original values.

        Unknown tokens (matching the token shape but not in the vault)
        are left untouched. Runs in a single regex pass, O(len(text)).
        """

        def _sub(match: re.Match[str]) -> str:
            entry = self._token_to_value.get(match.group(0))
            if entry is None:
                return match.group(0)
            return entry[1]

        return _TOKEN_RE.sub(_sub, text)
