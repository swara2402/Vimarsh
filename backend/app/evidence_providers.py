"""Credential-free Crossref DOI metadata adapter.

Only bibliographic metadata is retrieved. This does not fetch full-text content,
validate scientific claims, or establish that a cited result is correct.
"""
from __future__ import annotations

import re
from urllib.parse import quote

import httpx


_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_CROSSREF_WORKS = "https://api.crossref.org/works/"


def resolve_crossref_doi(doi: str) -> dict:
    normalized = doi.strip()
    if normalized.lower().startswith("https://doi.org/"):
        normalized = normalized[len("https://doi.org/"):]
    if normalized.lower().startswith("http://doi.org/"):
        normalized = normalized[len("http://doi.org/"):]
    if not _DOI_PATTERN.fullmatch(normalized):
        raise ValueError("A valid DOI is required")

    try:
        response = httpx.get(
            _CROSSREF_WORKS + quote(normalized, safe="/"),
            headers={"Accept": "application/json", "User-Agent": "VIMARSH/0.2"},
            timeout=8.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError("Crossref metadata service is temporarily unavailable") from exc

    if response.status_code == 404:
        raise LookupError("DOI was not found in Crossref")
    if response.status_code != 200:
        raise RuntimeError(f"Crossref returned HTTP {response.status_code}")

    try:
        message = response.json()["message"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Crossref returned an invalid metadata response") from exc

    if not isinstance(message, dict):
        raise RuntimeError("Crossref returned an invalid metadata response")

    titles = message.get("title") or []
    authors = message.get("author") or []
    published = message.get("published-print") or message.get("published-online") or message.get("created") or {}
    date_parts = published.get("date-parts") or []
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    author_names = [
        " ".join(part for part in (author.get("given"), author.get("family")) if part)
        for author in authors[:30]
    ]
    return {
        "provider": "Crossref",
        "doi": message.get("DOI", normalized),
        "title": str(titles[0])[:1000] if titles else None,
        "authors": author_names,
        "publisher": str(message.get("publisher", ""))[:500] or None,
        "year": year,
        "work_type": str(message.get("type", ""))[:100] or None,
        "container_title": str((message.get("container-title") or [""])[0])[:500] or None,
        "url": "https://doi.org/" + normalized,
        "metadata_only": True,
        "claim_verification": "not_performed",
    }
