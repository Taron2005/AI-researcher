"""
Planner-only tool: search arXiv and Semantic Scholar directly, so the
Planner cites real papers instead of relying on the model's own (possibly
stale or hallucinated) training-data recall — CLAUDE.md rule 2. Same two
APIs used by hand earlier in this project to verify claims (see DECISIONS.md
for a real example of this mattering: the GBS/"creativity level" check and
the MLE-bench benchmark verification both used exactly these two APIs).

Exposed to the LLM as one function-calling tool. The TOOL_SCHEMA is what the
model sees; search_papers() is the plain Python function the harness
actually runs once the model requests that tool call.
"""

import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()

# https, not http -- verified live that http://export.arxiv.org 301-redirects
# every request to https, so the old http:// URL paid for two full
# connections (the redirect, then the real request) instead of one. Real
# measured latency data (DECISIONS.md) showed arXiv's actual failures
# weren't slow responses -- successful calls were fast -- they were 429s and
# hangs from having ZERO throttling despite arXiv's own Terms of Use stating
# a hard limit of one request per 3 seconds, single connection at a time
# (info.arxiv.org/help/api/tou.html). A bigger timeout can't fix a rate
# limit, so this is the real fix, not another blind timeout bump.
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT_SECONDS = 25

# Optional -- raises Semantic Scholar's rate limit above the anonymous tier
# (which 429s quickly, per DECISIONS.md). Works fine without it, just tighter
# limits. Header name confirmed via Semantic Scholar's own API docs pattern:
# `x-api-key: YOUR_KEY` on every request.
_SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

# Semantic Scholar's own key-approval email states the limit (1 request/sec,
# cumulative across all endpoints) is the *caller's* responsibility to stay
# under -- it rejects bursts rather than smoothing them. This is a real
# module-level throttle, not per-instance, since search_papers() could be
# called repeatedly across a whole Planner session.
_MIN_SECONDS_BETWEEN_REQUESTS = 1.1
_last_semantic_scholar_request_at = 0.0

# arXiv's own documented minimum (info.arxiv.org/help/api/tou.html: "make no
# more than one request every three seconds") -- previously had NO throttle
# at all for arXiv, unlike Semantic Scholar above. 3.5s gives the same small
# safety margin over the documented minimum that Semantic Scholar's 1.1s
# gives over its 1/sec limit.
_MIN_SECONDS_BETWEEN_ARXIV_REQUESTS = 3.5
_last_arxiv_request_at = 0.0

# Temporary: this IP is currently under arXiv's own documented "excessive
# usage" block (info.arxiv.org/help/api/tou.html) from tonight's cumulative
# testing -- confirmed real, not a code bug (throttling/redirect/429-handling
# all verified working correctly, DECISIONS.md). Every call still burns the
# real ~25s timeout or a 429 round-trip for zero benefit while blocked.
# Flip back to True once arXiv responds normally again -- Semantic Scholar
# alone still provides real, on-topic grounding in the meantime.
ARXIV_ENABLED = False

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search arXiv and Semantic Scholar for real papers on a topic. "
            "Use this before citing anything as 'state of the art' or "
            "'best practice' in the blueprint — do not rely on memory alone. "
            "arXiv matches queries as an OR of individual words, not a phrase "
            "or semantic match, and it cannot tell a rare term from a common "
            "one — a query with even one generic word ('energy', 'network', "
            "'learning') gets swamped by unrelated papers using that word. Use "
            "1-2 genuinely distinctive keywords only, e.g. 'QM8 SchNet' (works) "
            "rather than 'QM8 excitation energy GNN' (the word 'energy' alone "
            "returns Dark Energy Survey papers ahead of anything relevant)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "2-4 distinctive keywords, e.g. 'QM8 SchNet' — not a long natural-language phrase",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results per source (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def _search_arxiv(query: str, max_results: int) -> list[dict]:
    global _last_arxiv_request_at
    elapsed = time.monotonic() - _last_arxiv_request_at
    if elapsed < _MIN_SECONDS_BETWEEN_ARXIV_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_ARXIV_REQUESTS - elapsed)
    _last_arxiv_request_at = time.monotonic()

    response = requests.get(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            # arXiv's API defaults to sorting by submission date, not
            # relevance -- without this, results are recent-but-irrelevant
            # rather than actually matching the query. Found by testing
            # (see DECISIONS.md), not documented prominently by arXiv itself.
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        # Same graceful-degradation pattern as Semantic Scholar's 429 branch
        # below -- previously arXiv had no equivalent, so a rate limit here
        # surfaced as a generic requests.HTTPError string instead of a clear
        # "this is a rate limit" signal.
        return [{"source": "arxiv", "error": "rate-limited (429)"}]
    response.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    return [
        {
            "source": "arxiv",
            "title": entry.findtext("atom:title", default="", namespaces=ns).strip(),
            "url": entry.findtext("atom:id", default="", namespaces=ns).strip(),
            "summary": entry.findtext("atom:summary", default="", namespaces=ns).strip()[:500],
        }
        for entry in root.findall("atom:entry", ns)
    ]


def _search_semantic_scholar(query: str, max_results: int) -> list[dict]:
    global _last_semantic_scholar_request_at
    elapsed = time.monotonic() - _last_semantic_scholar_request_at
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_semantic_scholar_request_at = time.monotonic()

    headers = {"x-api-key": _SEMANTIC_SCHOLAR_API_KEY} if _SEMANTIC_SCHOLAR_API_KEY else {}
    response = requests.get(
        SEMANTIC_SCHOLAR_API,
        params={
            "query": query,
            "limit": max_results,
            "fields": "title,url,abstract,citationCount,year",
        },
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        # Anonymous rate limit (no API key configured) -- degrade gracefully
        # rather than fail the whole tool call; arXiv results may still be usable.
        return [{"source": "semantic_scholar", "error": "rate-limited (429)"}]
    response.raise_for_status()

    return [
        {
            "source": "semantic_scholar",
            "title": paper.get("title"),
            "url": paper.get("url"),
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount"),
            "summary": (paper.get("abstract") or "")[:500],
        }
        for paper in response.json().get("data", [])
    ]


def _active_sources() -> list[tuple]:
    sources = [(_search_arxiv, "arxiv")] if ARXIV_ENABLED else []
    sources.append((_search_semantic_scholar, "semantic_scholar"))
    return sources


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Called by the harness when the Planner's tool call names `search_papers`.
    Queries both sources independently -- a failure in one (e.g. Semantic
    Scholar's rate limit) doesn't lose the other's results.
    """
    results = []
    for search_fn, source_name in _active_sources():
        try:
            results.extend(search_fn(query, max_results))
        except Exception as e:
            # Broad on purpose: a network error (requests.RequestException)
            # isn't the only way a source can fail -- an outage can return
            # an HTML error page instead of XML/JSON, which fails parsing
            # (xml.etree.ParseError, json.JSONDecodeError) rather than the
            # request itself. One source's failure, whatever kind, should
            # never lose the other source's real results.
            results.append({"source": source_name, "error": str(e)})
    return results


def search_succeeded(results: list[dict]) -> bool:
    """
    True if at least one currently-active source responded without failing,
    whatever it found (including a real, clean zero-hit response). Owned
    here, not left for a caller to reimplement, because only this module
    knows how many sources are actually active right now -- a caller-side
    check hardcoded to "fewer than 2 errors" broke the moment ARXIV_ENABLED
    could be False, since a single failed source would then wrongly still
    count as fewer than 2 errors (DECISIONS.md).
    """
    error_count = sum(1 for r in results if "error" in r)
    return error_count < len(_active_sources())
