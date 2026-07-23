"""Minimal two-stage search and synthesis pipeline for the MVP MCP server."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


_PROVIDERS = ("exa", "tavily")
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class MVPSettings:
    exa_api_key: str | None = None
    exa_api_url: str = "https://api.exa.ai"
    tavily_api_key: str | None = None
    tavily_api_url: str = "https://api.tavily.com"
    synth_api_url: str | None = None
    synth_api_key: str | None = None
    synth_model: str | None = None
    cache_ttl_seconds: int = 1800
    cache_max_entries: int = 128
    raw_content_max_chars: int = 6000
    request_timeout_seconds: int = 90

    @classmethod
    def from_env(cls) -> "MVPSettings":
        return cls(
            exa_api_key=os.getenv("EXA_API_KEY"),
            exa_api_url=os.getenv("EXA_API_URL", "https://api.exa.ai"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            tavily_api_url=os.getenv("TAVILY_API_URL", "https://api.tavily.com"),
            synth_api_url=os.getenv("SYNTH_API_URL"),
            synth_api_key=os.getenv("SYNTH_API_KEY"),
            synth_model=os.getenv("SYNTH_MODEL"),
            cache_ttl_seconds=_env_int("SEARCH_CACHE_TTL_SECONDS", 1800, 60, 86400),
            cache_max_entries=_env_int("SEARCH_CACHE_MAX_ENTRIES", 128, 8, 2048),
            raw_content_max_chars=_env_int("SEARCH_RAW_CONTENT_MAX_CHARS", 6000, 500, 20000),
            request_timeout_seconds=_env_int("SEARCH_REQUEST_TIMEOUT_SECONDS", 90, 10, 300),
        )


class SearchCache:
    """Small in-memory TTL cache keyed by opaque search IDs."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 128,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def set(self, search_id: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self._purge_expired()
            self._items[search_id] = (self._clock() + self.ttl_seconds, copy.deepcopy(value))
            self._items.move_to_end(search_id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    async def get(self, search_id: str) -> dict[str, Any] | None:
        async with self._lock:
            self._purge_expired()
            cached = self._items.get(search_id)
            if cached is None:
                return None
            self._items.move_to_end(search_id)
            return copy.deepcopy(cached[1])

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [key for key, (expires_at, _) in self._items.items() if expires_at <= now]
        for key in expired:
            self._items.pop(key, None)


def canonicalize_url(url: str) -> str:
    """Remove fragments and common tracking parameters without discarding useful queries."""
    candidate = (url or "").strip()
    if not candidate:
        return ""
    try:
        parts = urlsplit(candidate)
        scheme = parts.scheme.lower()
        hostname = (parts.hostname or "").lower()
        if not scheme or not hostname:
            return candidate
        port = parts.port
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        ]
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((scheme, netloc, path, urlencode(sorted(query)), ""))
    except (ValueError, UnicodeError):
        return candidate


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _text_fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    if len(normalized) < 80:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _merge_text(existing: str, incoming: str, limit: int) -> str:
    existing = existing.strip()
    incoming = incoming.strip()
    if not incoming:
        return _bounded_text(existing, limit)
    if not existing:
        return _bounded_text(incoming, limit)
    normalized_existing = re.sub(r"\s+", " ", existing).lower()
    normalized_incoming = re.sub(r"\s+", " ", incoming).lower()
    if normalized_incoming in normalized_existing:
        return _bounded_text(existing, limit)
    if normalized_existing in normalized_incoming:
        return _bounded_text(incoming, limit)
    return _bounded_text(f"{existing}\n\n---\n\n{incoming}", limit)


def normalize_and_deduplicate(
    provider_results: dict[str, list[dict[str, Any]]],
    raw_content_max_chars: int = 6000,
) -> list[dict[str, Any]]:
    """Fuse provider-ranked results using exact URL/content deduplication and RRF."""
    fused: list[dict[str, Any]] = []
    url_index: dict[str, int] = {}
    content_index: dict[str, int] = {}

    for provider in _PROVIDERS:
        for rank, result in enumerate(provider_results.get(provider, []), start=1):
            url = canonicalize_url(str(result.get("url") or ""))
            if not url:
                continue
            excerpt = _bounded_text(result.get("excerpt") or result.get("content"), 1800)
            raw_content = _bounded_text(
                result.get("raw_content") or result.get("text") or excerpt,
                raw_content_max_chars,
            )
            fingerprint = _text_fingerprint(raw_content or excerpt)
            existing_index = url_index.get(url)
            if existing_index is None and fingerprint:
                existing_index = content_index.get(fingerprint)

            provider_score = result.get("provider_score")
            try:
                provider_score = float(provider_score) if provider_score is not None else None
            except (TypeError, ValueError):
                provider_score = None

            rrf_score = 1.0 / (60 + rank)
            if existing_index is None:
                item = {
                    "id": "",
                    "title": _bounded_text(result.get("title") or url, 500),
                    "url": url,
                    "alternate_urls": [],
                    "providers": [provider],
                    "published_date": result.get("published_date") or None,
                    "score": rrf_score,
                    "provider_scores": {provider: provider_score},
                    "excerpt": excerpt,
                    "raw_content": raw_content,
                }
                fused.append(item)
                existing_index = len(fused) - 1
            else:
                item = fused[existing_index]
                if provider not in item["providers"]:
                    item["providers"].append(provider)
                if url != item["url"] and url not in item["alternate_urls"]:
                    item["alternate_urls"].append(url)
                incoming_title = _bounded_text(result.get("title"), 500)
                if len(incoming_title) > len(item["title"]):
                    item["title"] = incoming_title
                item["published_date"] = item["published_date"] or result.get("published_date") or None
                item["excerpt"] = _merge_text(item["excerpt"], excerpt, 1800)
                item["raw_content"] = _merge_text(
                    item["raw_content"], raw_content, raw_content_max_chars
                )
                item["score"] += rrf_score
                item["provider_scores"][provider] = provider_score

            url_index[url] = existing_index
            if fingerprint:
                content_index[fingerprint] = existing_index

    fused.sort(key=lambda item: (-item["score"], item["url"]))
    for index, item in enumerate(fused, start=1):
        item["id"] = f"S{index}"
        item["score"] = round(item["score"], 6)
        item["provider_scores"] = {
            provider: score for provider, score in item["provider_scores"].items() if score is not None
        }
        if not item["alternate_urls"]:
            item.pop("alternate_urls")
    return fused


class SearchPipeline:
    def __init__(
        self,
        settings: MVPSettings | None = None,
        cache: SearchCache | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self.settings = settings or MVPSettings.from_env()
        self.cache = cache or SearchCache(
            ttl_seconds=self.settings.cache_ttl_seconds,
            max_entries=self.settings.cache_max_entries,
        )
        self._client_factory = client_factory

    async def search_sources(self, query: str, max_results: int = 6) -> dict[str, Any]:
        query = (query or "").strip()
        if not query or len(query) > 400:
            return self._error(
                "INVALID_QUERY",
                "query must contain 1-400 characters.",
                "Send a concise, self-contained web search query.",
            )
        if not 1 <= max_results <= 10:
            return self._error(
                "INVALID_MAX_RESULTS",
                "max_results must be between 1 and 10.",
                "Retry with max_results in the inclusive range 1-10.",
            )

        provider_status: dict[str, dict[str, Any]] = {}
        provider_results: dict[str, list[dict[str, Any]]] = {name: [] for name in _PROVIDERS}
        scheduled: list[tuple[str, Any]] = []

        if self.settings.exa_api_key:
            scheduled.append(("exa", self._search_exa(query, max_results)))
        else:
            provider_status["exa"] = {
                "status": "not_configured",
                "result_count": 0,
                "error": "EXA_API_KEY is not configured.",
            }
        if self.settings.tavily_api_key:
            scheduled.append(("tavily", self._search_tavily(query, max_results)))
        else:
            provider_status["tavily"] = {
                "status": "not_configured",
                "result_count": 0,
                "error": "TAVILY_API_KEY is not configured.",
            }

        if scheduled:
            outcomes = await asyncio.gather(
                *(operation for _, operation in scheduled), return_exceptions=True
            )
            for (provider, _), outcome in zip(scheduled, outcomes):
                if isinstance(outcome, BaseException):
                    provider_status[provider] = {
                        "status": "error",
                        "result_count": 0,
                        "error": _bounded_text(str(outcome), 500),
                    }
                else:
                    provider_results[provider] = outcome
                    provider_status[provider] = {
                        "status": "ok",
                        "result_count": len(outcome),
                        "error": None,
                    }

        sources = normalize_and_deduplicate(
            provider_results, raw_content_max_chars=self.settings.raw_content_max_chars
        )[: max_results * 2]
        successful_providers = [
            provider for provider in _PROVIDERS if provider_status[provider]["status"] == "ok"
        ]
        if not sources:
            return {
                "ok": False,
                "status": "failed",
                "search_id": None,
                "query": query,
                "provider_status": provider_status,
                "source_count": 0,
                "sources": [],
                "error": {
                    "code": "NO_SEARCH_RESULTS",
                    "message": "Neither configured provider returned usable search results.",
                    "resolution": "Check provider configuration or retry search_sources with a clearer query.",
                    "retryable": True,
                },
            }

        search_id = f"search_{secrets.token_urlsafe(12)}"
        created_at = datetime.now(timezone.utc).isoformat()
        cached = {
            "search_id": search_id,
            "query": query,
            "created_at": created_at,
            "provider_status": provider_status,
            "sources": sources,
        }
        await self.cache.set(search_id, cached)
        complete = len(successful_providers) == len(_PROVIDERS)
        return {
            "ok": True,
            "status": "complete" if complete else "partial",
            **cached,
            "source_count": len(sources),
            "cache_expires_in_seconds": self.settings.cache_ttl_seconds,
            "next_step": (
                "Call GrokSearchMVP:synthesize_search with this search_id to produce the final "
                "structured answer. If the user only asked for links or raw evidence, these sources "
                "can be used directly."
            ),
        }

    async def synthesize_search(self, search_id: str, focus: str = "") -> dict[str, Any]:
        search_id = (search_id or "").strip()
        cached = await self.cache.get(search_id)
        if cached is None:
            return self._error(
                "SEARCH_ID_NOT_FOUND_OR_EXPIRED",
                "The search_id is unknown or its cached evidence has expired.",
                "Call GrokSearchMVP:search_sources again, then pass the new search_id here.",
                search_id=search_id,
            )

        missing = [
            name
            for name, value in (
                ("SYNTH_API_URL", self.settings.synth_api_url),
                ("SYNTH_API_KEY", self.settings.synth_api_key),
                ("SYNTH_MODEL", self.settings.synth_model),
            )
            if not value
        ]
        if missing:
            return {
                **self._error(
                    "SYNTHESIS_NOT_CONFIGURED",
                    f"Missing synthesis configuration: {', '.join(missing)}.",
                    "Configure the missing variables and retry synthesize_search with the same search_id before it expires.",
                    search_id=search_id,
                ),
                "query": cached["query"],
                "sources": cached["sources"],
            }

        focus = _bounded_text(focus, 1000)
        prompt = self._build_synthesis_prompt(cached, focus)
        try:
            raw_output = await self._call_synthesis(prompt)
        except Exception as exc:
            return {
                **self._error(
                    "SYNTHESIS_REQUEST_FAILED",
                    _bounded_text(str(exc), 700),
                    "Retry synthesize_search with the same search_id. The cached source evidence is included below.",
                    search_id=search_id,
                ),
                "query": cached["query"],
                "sources": cached["sources"],
            }

        parsed, parse_warning = _parse_json_object(raw_output)
        normalized = _normalize_synthesis(parsed, raw_output, cached["sources"])
        warnings = normalized.pop("validation_warnings")
        if parse_warning:
            warnings.insert(0, parse_warning)
        return {
            "ok": True,
            "status": "complete" if not warnings else "complete_with_warnings",
            "search_id": search_id,
            "query": cached["query"],
            "focus": focus or None,
            **normalized,
            "validation_warnings": warnings,
            "sources": cached["sources"],
        }

    async def _search_exa(self, query: str, max_results: int) -> list[dict[str, Any]]:
        endpoint = f"{self.settings.exa_api_url.rstrip('/')}/search"
        headers = {
            "x-api-key": self.settings.exa_api_key or "",
            "Content-Type": "application/json",
        }
        body = {
            "query": query,
            "type": "auto",
            "numResults": max_results,
            "contents": {"text": {"maxCharacters": self.settings.raw_content_max_chars}},
        }
        async with self._client_factory(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        results = []
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or ""
            highlights = item.get("highlights") or []
            excerpt = "\n".join(str(value) for value in highlights if value) or text
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or item.get("id") or "",
                    "published_date": item.get("publishedDate") or None,
                    "provider_score": item.get("score"),
                    "excerpt": excerpt,
                    "raw_content": text,
                }
            )
        return results

    async def _search_tavily(self, query: str, max_results: int) -> list[dict[str, Any]]:
        endpoint = f"{self.settings.tavily_api_url.rstrip('/')}/search"
        headers = {
            "Authorization": f"Bearer {self.settings.tavily_api_key or ''}",
            "Content-Type": "application/json",
        }
        body = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": "text",
        }
        async with self._client_factory(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        results = []
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "published_date": item.get("published_date") or None,
                    "provider_score": item.get("score"),
                    "excerpt": item.get("content") or "",
                    "raw_content": item.get("raw_content") or item.get("content") or "",
                }
            )
        return results

    def _build_synthesis_prompt(self, cached: dict[str, Any], focus: str) -> str:
        evidence = json.dumps(cached["sources"], ensure_ascii=False, indent=2)
        return f"""You are the synthesis stage of a web-search pipeline.

Use only the supplied evidence. Preserve names, dates, numbers, qualifications, URLs, and disagreements. Deduplicate repeated facts, but do not silently discard unique details. Never invent missing information. Every key finding and conflict must cite one or more source IDs such as S1 or S3.

User query: {cached['query']}
Optional focus: {focus or 'None; answer the original query.'}

Return JSON only with exactly this top-level shape:
{{
  "summary": "clear answer that keeps important qualifications",
  "key_findings": [
    {{"title": "short finding title", "details": "full finding", "source_ids": ["S1"]}}
  ],
  "conflicts": [
    {{"topic": "what conflicts", "details": "how the sources differ", "source_ids": ["S2", "S4"]}}
  ]
}}

Evidence:
{evidence}
"""

    async def _call_synthesis(self, prompt: str) -> str:
        base_url = (self.settings.synth_api_url or "").rstrip("/")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.synth_api_key or ''}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.settings.synth_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Synthesize supplied web evidence faithfully. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        async with self._client_factory(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            if response.status_code in (400, 422):
                body.pop("response_format", None)
                response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or item.get("content") or "")
                for item in content
                if isinstance(item, dict)
            )
        content = str(content or "").strip()
        if not content:
            raise ValueError("Synthesis API returned an empty message.")
        return content

    @staticmethod
    def _error(
        code: str,
        message: str,
        resolution: str,
        search_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "error",
            "search_id": search_id,
            "error": {
                "code": code,
                "message": message,
                "resolution": resolution,
                "retryable": code not in {"INVALID_QUERY", "INVALID_MAX_RESULTS"},
            },
        }


def _parse_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = (raw_text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
            if isinstance(parsed, dict):
                return parsed, "Model output contained extra text around the JSON object."
        except json.JSONDecodeError:
            pass
    return None, "Model output was not valid JSON; the unparsed answer is preserved in summary."


def _coerce_source_ids(value: Any, valid_ids: set[str]) -> list[str]:
    if isinstance(value, str):
        candidates = re.findall(r"S\d+", value.upper())
    elif isinstance(value, list):
        candidates = [str(item).upper() for item in value]
    else:
        candidates = []
    return list(dict.fromkeys(item for item in candidates if item in valid_ids))


def _normalize_synthesis(
    payload: dict[str, Any] | None,
    raw_text: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_ids = {source["id"] for source in sources}
    warnings: list[str] = []
    if payload is None:
        return {
            "summary": _bounded_text(raw_text, 12000),
            "key_findings": [],
            "conflicts": [],
            "validation_warnings": warnings,
        }

    summary = _bounded_text(payload.get("summary") or payload.get("answer"), 12000)
    findings = []
    for index, item in enumerate(payload.get("key_findings") or payload.get("findings") or []):
        if not isinstance(item, dict):
            continue
        source_ids = _coerce_source_ids(item.get("source_ids") or item.get("sources"), valid_ids)
        details = _bounded_text(item.get("details") or item.get("content") or item.get("summary"), 5000)
        if not details:
            continue
        if not source_ids:
            warnings.append(f"Dropped key finding {index + 1} because it had no valid source IDs.")
            continue
        findings.append(
            {
                "title": _bounded_text(item.get("title") or item.get("claim") or f"Finding {index + 1}", 300),
                "details": details,
                "source_ids": source_ids,
            }
        )

    conflicts = []
    for index, item in enumerate(payload.get("conflicts") or []):
        if not isinstance(item, dict):
            continue
        source_ids = _coerce_source_ids(item.get("source_ids") or item.get("sources"), valid_ids)
        details = _bounded_text(item.get("details") or item.get("description") or item.get("conflict"), 5000)
        if not details:
            continue
        if not source_ids:
            warnings.append(f"Dropped conflict {index + 1} because it had no valid source IDs.")
            continue
        conflicts.append(
            {
                "topic": _bounded_text(item.get("topic") or item.get("title") or f"Conflict {index + 1}", 300),
                "details": details,
                "source_ids": source_ids,
            }
        )

    if not summary:
        warnings.append("The model returned no summary.")
    return {
        "summary": summary,
        "key_findings": findings,
        "conflicts": conflicts,
        "validation_warnings": warnings,
    }
