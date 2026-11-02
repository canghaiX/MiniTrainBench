from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

GSM8K_SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    marked = _ANSWER_RE.findall(text)
    candidates = marked or _NUMBER_RE.findall(text)
    return candidates[-1].replace(",", "") if candidates else None


def gsm8k_exact_match(response: str, target: str) -> float:
    actual = extract_gsm8k_answer(response)
    expected = extract_gsm8k_answer(target) or target.strip().replace(",", "")
    if actual is None:
        return 0.0
    try:
        return float(Decimal(actual) == Decimal(expected))
    except InvalidOperation:
        return 0.0
