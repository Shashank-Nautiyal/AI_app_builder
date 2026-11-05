from __future__ import annotations

import re

try:
    from json_repair import repair_json as _repair_with_json_repair
except ImportError:  # pragma: no cover - optional dependency
    _repair_with_json_repair = None


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


def _extract_json_candidate(text: str) -> str:
    if not text:
        return ""

    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_candidates:
        return text.strip()

    start = min(start_candidates)
    end_candidates = [idx for idx in (text.rfind("}"), text.rfind("]")) if idx != -1 and idx >= start]
    if not end_candidates:
        return text[start:].strip()

    end = max(end_candidates)
    return text[start : end + 1].strip()


def _count_unescaped_quotes(text: str) -> int:
    count = 0
    escaped = False
    for char in text:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            count += 1
        escaped = False
    return count


def repair_structured_json(text: str) -> str:
    cleaned = _extract_json_candidate(_strip_code_fences(text))
    if not cleaned:
        return cleaned

    if _repair_with_json_repair is not None:
        try:
            repaired = _repair_with_json_repair(cleaned)
            if repaired:
                return repaired
        except Exception:
            pass

    repaired = cleaned
    if _count_unescaped_quotes(repaired) % 2 == 1:
        repaired += '"'

    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)

    open_braces = repaired.count("{") - repaired.count("}")
    if open_braces > 0:
        repaired += "}" * open_braces

    open_brackets = repaired.count("[") - repaired.count("]")
    if open_brackets > 0:
        repaired += "]" * open_brackets

    return repaired
