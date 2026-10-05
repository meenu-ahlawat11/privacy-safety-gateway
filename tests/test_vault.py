"""Tests for Vault. Synthetic data only."""

from app.vault import Vault


def test_same_value_gives_same_token() -> None:
    v = Vault()
    assert v.tokenize("EMAIL", "alice@example.com") == v.tokenize("EMAIL", "alice@example.com")


def test_different_values_give_different_numbers() -> None:
    v = Vault()
    t1 = v.tokenize("EMAIL", "alice@example.com")
    t2 = v.tokenize("EMAIL", "bob@example.com")
    assert t1 == "<EMAIL_1>"
    assert t2 == "<EMAIL_2>"


def test_counters_are_per_type() -> None:
    v = Vault()
    assert v.tokenize("EMAIL", "alice@example.com") == "<EMAIL_1>"
    assert v.tokenize("PHONE", "9876543210") == "<PHONE_1>"
    assert v.tokenize("EMAIL", "bob@example.com") == "<EMAIL_2>"
    assert v.tokenize("PHONE", "9123456780") == "<PHONE_2>"


def test_restore_round_trip() -> None:
    v = Vault()
    v.tokenize("EMAIL", "alice@example.com")
    v.tokenize("PHONE", "9876543210")
    text = "Mail <EMAIL_1> or call <PHONE_1> today"
    assert v.restore(text) == "Mail alice@example.com or call 9876543210 today"


def test_restore_leaves_unknown_tokens() -> None:
    v = Vault()
    v.tokenize("EMAIL", "alice@example.com")
    text = "Use <EMAIL_1> and <FOO_9>"
    assert v.restore(text) == "Use alice@example.com and <FOO_9>"


def test_email_10_does_not_corrupt_email_1() -> None:
    v = Vault()
    tokens = [v.tokenize("EMAIL", f"user{i}@example.com") for i in range(10)]
    assert tokens[0] == "<EMAIL_1>"
    assert tokens[9] == "<EMAIL_10>"
    text = "first <EMAIL_1> last <EMAIL_10> mid <EMAIL_1>"
    assert v.restore(text) == (
        "first user0@example.com last user9@example.com mid user0@example.com"
    )
