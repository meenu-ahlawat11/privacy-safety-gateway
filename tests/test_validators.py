"""Tests for Luhn and Verhoeff validators. Synthetic data only."""

from app.detectors.validators import (
    luhn_valid,
    verhoeff_check_digit,
    verhoeff_valid,
)


def test_luhn_valid_card_passes() -> None:
    assert luhn_valid("4111111111111111") is True


def test_luhn_invalid_card_fails() -> None:
    assert luhn_valid("4111111111111112") is False


def test_luhn_allows_spaces() -> None:
    assert luhn_valid("4111 1111 1111 1111") is True


def test_luhn_short_number_fails() -> None:
    assert luhn_valid("411111") is False


def _make_fake_aadhaar() -> str:
    prefix = "23456789012"  # 11 digits, starts with 2-9
    check = verhoeff_check_digit(prefix)
    return prefix + str(check)


def test_verhoeff_generated_aadhaar_passes() -> None:
    aadhaar = _make_fake_aadhaar()
    assert len(aadhaar) == 12
    assert verhoeff_valid(aadhaar) is True


def test_verhoeff_changed_digit_fails() -> None:
    aadhaar = _make_fake_aadhaar()
    replacement = "5" if aadhaar[0] != "5" else "6"
    corrupted = replacement + aadhaar[1:]
    assert verhoeff_valid(corrupted) is False


def test_verhoeff_adjacent_swap_fails() -> None:
    aadhaar = _make_fake_aadhaar()
    assert aadhaar[0] != aadhaar[1]
    swapped = aadhaar[1] + aadhaar[0] + aadhaar[2:]
    assert verhoeff_valid(swapped) is False
