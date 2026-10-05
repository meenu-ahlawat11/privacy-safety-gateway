"""Benchmark KeywordDetector (trie) vs a naive str.find baseline.

Usage: python benchmarks/bench_trie.py
Writes benchmarks/results.csv and benchmarks/results.png.
"""

import csv
import random
import string
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detectors.keyword_trie import KeywordDetector, Trie  # noqa: E402

SEED = 42
TEXT_LEN = 20_000
SIZES = [10, 100, 1000, 5000, 10000]
REPEATS = 5


def _random_word(rng: random.Random, min_len: int, max_len: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(min_len, max_len)))


def build_keywords(rng: random.Random, n: int) -> list[str]:
    keywords: set[str] = set()
    while len(keywords) < n:
        keywords.add(_random_word(rng, 4, 12))
    return sorted(keywords)


def build_text(rng: random.Random, keywords: list[str]) -> str:
    words: list[str] = []
    total = 0
    i = 0
    while total < TEXT_LEN:
        if i % 50 == 25 and keywords:
            word = rng.choice(keywords)
        else:
            word = _random_word(rng, 1, 10)
        words.append(word)
        total += len(word) + 1
        i += 1
    return " ".join(words)[:TEXT_LEN]


def _is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit()


def naive_find_spans(text: str, keywords: list[str]) -> set[tuple[int, int]]:
    """Lowercase once, str.find loop per keyword, same whole-word rule.

    At each start position only the longest match is kept, mirroring the
    trie detector's semantics.
    """
    lowered = text.lower()
    best: dict[int, int] = {}
    for keyword in keywords:
        needle = keyword.lower()
        if not needle:
            continue
        pos = lowered.find(needle)
        while pos != -1:
            end = pos + len(needle)
            left_ok = pos == 0 or not _is_word_char(text[pos - 1])
            right_ok = end == len(text) or not _is_word_char(text[end])
            if left_ok and right_ok:
                if pos not in best or end > best[pos]:
                    best[pos] = end
            pos = lowered.find(needle, pos + 1)
    return {(start, end) for start, end in best.items()}


def trie_find_spans(trie: Trie, text: str) -> set[tuple[int, int]]:
    return set(trie.find_all(text))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def time_call(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return median(samples)


def main() -> None:
    rng = random.Random(SEED)
    all_keywords = build_keywords(rng, max(SIZES))
    text = build_text(rng, all_keywords)

    rows = []
    for size in SIZES:
        keywords = all_keywords[:size]

        trie = Trie()

        def make_trie() -> Trie:
            t = Trie()
            for kw in keywords:
                t.insert(kw)
            return t

        build_samples = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            fresh = make_trie()
            build_samples.append((time.perf_counter() - start) * 1000.0)
        build_ms = median(build_samples)
        trie = fresh

        trie_search_ms = time_call(lambda: trie_find_spans(trie, text), REPEATS)
        naive_search_ms = time_call(lambda: naive_find_spans(text, keywords), REPEATS)

        trie_spans = trie_find_spans(trie, text)
        naive_spans = naive_find_spans(text, keywords)
        assert trie_spans == naive_spans, (
            f"span mismatch at size {size}: "
            f"{len(trie_spans ^ naive_spans)} differing spans"
        )

        rows.append((size, build_ms, trie_search_ms, naive_search_ms))
        print(f"{size:>6}  {build_ms:>12.2f}  {trie_search_ms:>14.2f}  {naive_search_ms:>14.2f}")

    header = ("keywords", "trie_build_ms", "trie_search_ms", "naive_search_ms")
    print("\nkeywords | trie build ms | trie search ms | naive search ms")
    print("-" * 64)
    for size, b, t, n in rows:
        print(f"{size:>8} | {b:>13.2f} | {t:>14.2f} | {n:>15.2f}")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    sizes = [r[0] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [r[2] for r in rows], marker="o", label="trie search")
    plt.plot(sizes, [r[3] for r in rows], marker="s", label="naive search")
    plt.xscale("log")
    plt.xlabel("keyword count")
    plt.ylabel("search time (ms)")
    plt.title("KeywordDetector (trie) vs naive str.find")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=120)

    print(f"\nWrote {csv_path} and {out_dir / 'results.png'}")


if __name__ == "__main__":
    main()
