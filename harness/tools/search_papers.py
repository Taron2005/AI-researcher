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

ARXIV_API = "http://export.arxiv.org/api/query"
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


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Called by the harness when the Planner's tool call names `search_papers`.
    Queries both sources independently -- a failure in one (e.g. Semantic
    Scholar's rate limit) doesn't lose the other's results.
    """
    results = []
    for search_fn, source_name in [
        (_search_arxiv, "arxiv"),
        (_search_semantic_scholar, "semantic_scholar"),
    ]:
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
