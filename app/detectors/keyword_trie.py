"""Keyword detection via a hand-written trie (prefix tree).

A trie stores each keyword character-by-character along root-to-leaf
paths. Words sharing a prefix share nodes, so matching many keywords
against a text costs roughly O(N * M), where N is the text length and
M is the maximum keyword length — independent of the number of
keywords. Insert is O(L) per word of length L; space is O(total
characters across all keywords).
"""

from app.detectors.base import Detection, Detector


class TrieNode:
    __slots__ = ("children", "is_end", "word")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.word: str | None = None


def _is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit()


class Trie:
    """A case-insensitive trie of keywords."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        if not word:
            return
        node = self.root
        for ch in word.lower():
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        node.word = word.lower()

    def find_all(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) spans of whole-word keyword matches.

        At each start position only the longest matching keyword is
        reported. A match must not be immediately preceded or followed
        by a letter or digit.
        """
        if not text:
            return []
        lowered = text.lower()
        spans: list[tuple[int, int]] = []
        for i in range(len(text)):
            node = self.root
            best: int | None = None
            for j in range(i, len(text)):
                child = node.children.get(lowered[j])
                if child is None:
                    break
                node = child
                if node.is_end:
                    end = j + 1
                    left_ok = i == 0 or not _is_word_char(text[i - 1])
                    right_ok = end == len(text) or not _is_word_char(text[end])
                    if left_ok and right_ok:
                        best = end  # keep the longest; later ends win
            if best is not None:
                spans.append((i, best))
        return spans


class KeywordDetector(Detector):
    """Detects whole-word occurrences of a fixed keyword list."""

    def __init__(self, keywords: list[str]) -> None:
        self._trie = Trie()
        for keyword in keywords:
            self._trie.insert(keyword)

    def detect(self, text: str) -> list[Detection]:
        return [
            Detection(
                start=start,
                end=end,
                type="KEYWORD",
                confidence=0.90,
                source="keyword_trie",
            )
            for start, end in self._trie.find_all(text)
        ]
