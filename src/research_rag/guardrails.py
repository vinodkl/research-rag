"""Deterministic input, evidence, and output checks for the RAG pipeline.

These cheap rules are a first layer, not a complete moderation system.
"""

import math
import re
import unicodedata
from dataclasses import dataclass

from research_rag.models import Chunk

# Common prompt-injection phrasings and role-token attacks.
INJECTION = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:the\s+|all\s+)?(?:previous|prior|above)\s+"
    r"(?:instructions?|prompts?)|"
    r"(?:reveal|show|print|repeat)\s+(?:your\s+)?(?:system|developer)\s+(?:prompt|instructions?)|"
    r"\byou\s+are\s+now\b|\b(?:override|bypass)\s+(?:the\s+)?(?:rules?|guardrails?)\b|"
    r"<\|\s*(?:system|assistant|developer)\s*\|>|\[\s*(?:system|developer)\s*\]",
    re.IGNORECASE | re.DOTALL,
)

# Intent patterns still allow papers that merely discuss safety or attacks.
UNSAFE_REQUEST = re.compile(
    r"\b(?:write|generate|produce|give me)\b.{0,50}\b(?:hate speech|racial slurs?|"
    r"pornographic|nsfw)\b|"
    r"\b(?:how (?:can|do) i|instructions? (?:for|to))\b.{0,50}\b(?:kill|harm|attack)\s+"
    r"(?:a person|someone|people)\b|"
    r"\b(?:you are|you(?:'re| are))\s+(?:an?\s+)?(?:idiot|stupid|worthless)\b",
    re.IGNORECASE | re.DOTALL,
)

# Ambiguous technical questions continue to the retrieval evidence guard.
OFF_TOPIC = re.compile(
    r"\b(?:best|nearest|recommend)\s+(?:pizza|restaurant|cafe)\b|"
    r"\b(?:pizza|cooking|dinner)\s+recipes?\b|\brestaurant\s+recommendations?\b|"
    r"\b(?:weather|temperature)\s+(?:today|tomorrow|in\s+[A-Za-z])|"
    r"\b(?:book|find)\s+(?:me\s+)?(?:a\s+)?(?:flight|hotel)\b|"
    r"\b(?:recipe\s+for|daily\s+horoscope|sports?\s+scores?)\b",
    re.IGNORECASE,
)

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
API_TOKEN = re.compile(r"(?<!\w)(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}(?!\w)")
PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-])\d{3}[ .-]\d{4}(?!\w)"
)
CREDIT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

MIN_SCORE = 0.10  # cosine similarity below this means "nothing relevant exists"
MAX_CITATION_QUOTE_CHARS = 300


@dataclass(frozen=True)
class SanitizedQuestion:
    """Text safe to send downstream, without retaining the original PII."""

    text: str
    redactions: tuple[str, ...] = ()


def normalize(text: str) -> str:
    """Normalize confusable formatting before applying rules or retrieval."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def check_question(question: str) -> str | None:
    """Returns a refusal message, or None if the question may proceed."""
    question = normalize(question)
    if not 3 <= len(question) <= 500:
        return "Please ask a question between 3 and 500 characters."
    if INJECTION.search(question):
        return (
            "That looks like an attempt to change my instructions, so I did not run it."
        )
    if UNSAFE_REQUEST.search(question):
        return "I cannot help with that request."
    if OFF_TOPIC.search(question):
        return "I can only answer questions about the indexed machine-learning papers."
    return None


def sanitize_question(question: str) -> SanitizedQuestion:
    """Replace PII without retaining the original values in logs or traces."""
    text = normalize(question)
    redactions: list[str] = []

    def replace(pattern: re.Pattern, kind: str, value: str) -> str:
        counter = 0

        def replacement(match: re.Match) -> str:
            nonlocal counter
            if kind == "CREDIT_CARD" and not _valid_card(match.group(0)):
                return match.group(0)
            counter += 1
            redactions.append(kind)
            return f"[{kind}_{counter}]"

        return pattern.sub(replacement, value)

    for pattern, kind in (
        (API_TOKEN, "API_TOKEN"),
        (CREDIT_CARD, "CREDIT_CARD"),
        (EMAIL, "EMAIL"),
        (PHONE, "PHONE"),
    ):
        text = replace(pattern, kind, text)
    return SanitizedQuestion(text=text, redactions=tuple(redactions))


def check_retrieval(results: list[tuple[Chunk, float]]) -> str | None:
    """A vector store always returns k results - even for nonsense. The score
    floor turns 'best of nothing' into an honest refusal instead of a guess."""
    if not any(math.isfinite(score) and score >= MIN_SCORE for _, score in results):
        return "I could not find anything in the indexed papers about that."
    return None


def check_output(answer: str) -> str | None:
    """Block empty responses or sensitive values produced by the model."""
    if not isinstance(answer, str) or not answer.strip():
        return "I could not produce a usable answer from the indexed papers."
    if sanitize_question(answer).redactions:
        return "I withheld the answer because it may contain sensitive personal information."
    return None


def check_citations(
    citations: list[dict], results: list[tuple[Chunk, float]]
) -> list[dict]:
    """Keep only citations whose quote appears verbatim in the chunk they cite."""
    if not isinstance(citations, list):
        return []
    by_id = {chunk.id: chunk for chunk, _ in results}
    valid = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        chunk_id = citation.get("chunk_id")
        quote = citation.get("quote")
        if not isinstance(chunk_id, str) or not isinstance(quote, str):
            continue

        chunk = by_id.get(chunk_id)
        normalized_quote = _squash(quote)
        key = (chunk_id, normalized_quote)
        if (
            chunk is None
            or not 10 < len(normalized_quote) <= MAX_CITATION_QUOTE_CHARS
            or len(quote) > MAX_CITATION_QUOTE_CHARS
            or normalized_quote not in _squash(chunk.text)
            or sanitize_question(quote).redactions
            or key in seen
        ):
            continue

        seen.add(key)
        valid.append(
            {
                "chunk_id": chunk.id,
                "quote": quote,
                "source": f"{chunk.title}, {chunk.section}, p.{chunk.page}",
            }
        )
    return valid


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _valid_card(value: str) -> bool:
    """Luhn check avoids treating arbitrary long research numbers as cards."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
