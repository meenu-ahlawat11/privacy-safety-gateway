"""Tests for Trie and KeywordDetector. Synthetic data only."""

from app.detectors.keyword_trie import KeywordDetector, Trie


def test_single_match_exact_span() -> None:
    d = KeywordDetector(["falcon"])
    text = "the falcon flies"
    detections = d.detect(text)
    assert len(detections) == 1
    assert (detections[0].start, detections[0].end) == (4, 10)
    assert text[detections[0].start : detections[0].end] == "falcon"
    assert detections[0].type == "KEYWORD"
    assert detections[0].confidence == 0.90
    assert detections[0].source == "keyword_trie"


def test_case_insensitive_match() -> None:
    d = KeywordDetector(["project falcon"])
    detections = d.detect("Brief on PROJECT FALCON today")
    assert len(detections) == 1
    assert detections[0].start == 9
    assert detections[0].end == 23


def test_multi_word_keyword() -> None:
    d = KeywordDetector(["project falcon"])
    text = "deploy project falcon soon"
    detections = d.detect(text)
    assert len(detections) == 1
    assert text[detections[0].start : detections[0].end] == "project falcon"


def test_no_match() -> None:
    d = KeywordDetector(["falcon"])
    assert d.detect("nothing sensitive here") == []


def test_empty_keyword_list() -> None:
    d = KeywordDetector([])
    assert d.detect("project falcon") == []


def test_empty_text() -> None:
    d = KeywordDetector(["falcon"])
    assert d.detect("") == []


def test_whole_word_rule_no_match_inside_longer_word() -> None:
    d = KeywordDetector(["cat"])
    assert d.detect("concatenate the cats") == []


def test_longest_match_preferred() -> None:
    d = KeywordDetector(["project", "project falcon"])
    text = "launch project falcon now"
    detections = d.detect(text)
    assert len(detections) == 1
    assert text[detections[0].start : detections[0].end] == "project falcon"


def test_two_keywords_in_one_text() -> None:
    d = KeywordDetector(["falcon", "phoenix"])
    text = "falcon and phoenix"
    detections = d.detect(text)
    assert [text[x.start : x.end] for x in detections] == ["falcon", "phoenix"]


def test_keyword_at_very_start_and_end() -> None:
    d = KeywordDetector(["falcon"])
    text = "falcon rises like a falcon"
    detections = d.detect(text)
    assert [(x.start, x.end) for x in detections] == [(0, 6), (20, 26)]


def test_punctuation_next_to_keyword() -> None:
    d = KeywordDetector(["falcon"])
    text = "(falcon), then falcon."
    detections = d.detect(text)
    assert [text[x.start : x.end] for x in detections] == ["falcon", "falcon"]


def test_trie_insert_and_find_directly() -> None:
    trie = Trie()
    trie.insert("falcon")
    trie.insert("fallen")
    assert trie.find_all("falcon fallen") == [(0, 6), (7, 13)]
