"""Brave Search Reddit backend for last30days skill.

Uses Brave Web Search API with site:reddit.com filter to find Reddit threads.
Replaces OpenAI Responses API (`web_search` tool) which requires direct OpenAI access.

No LLM dependency — Brave relevance ranking + score.py weighting is sufficient.
"""

import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

from . import http

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Depth-dependent result counts
DEPTH_COUNT = {"quick": 10, "default": 20, "deep": 30}


def search_reddit(
    api_key: str,
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> List[Dict[str, Any]]:
    """Search Reddit via Brave Search API with site:reddit.com filter.

    Runs 2-3 query variants to compensate for Brave's weaker semantic search
    compared to OpenAI's web_search tool.

    Args:
        api_key: Brave Search API key
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'

    Returns:
        List of raw item dicts compatible with openai_reddit.parse_reddit_response() output format:
        [{id, title, url, subreddit, date, why_relevant, relevance}, ...]
    """
    count = DEPTH_COUNT.get(depth, 20)
    core = _extract_core_subject(topic)

    # Generate query variants
    queries = _build_query_variants(core, topic)

    # Calculate freshness
    from .brave_search import _days_between, _brave_freshness
    days = _days_between(from_date, to_date)
    freshness = _brave_freshness(days)

    # Run all query variants, collect results
    all_results = []
    seen_urls = set()

    for i, query in enumerate(queries):
        try:
            results = _search_brave(query, api_key, count, freshness)
            for item in results:
                url = item.get("url", "")
                # Normalize URL for dedup
                norm_url = _normalize_reddit_url(url)
                if norm_url and norm_url not in seen_urls:
                    seen_urls.add(norm_url)
                    all_results.append(item)
        except http.HTTPError as e:
            sys.stderr.write(f"[Brave Reddit] Query {i+1} error: {e}\n")
            sys.stderr.flush()
            # Continue with remaining queries

    # Parse Brave results into Reddit item format
    items = _parse_results(all_results)

    sys.stderr.write(f"[Brave Reddit] {len(items)} Reddit threads found\n")
    sys.stderr.flush()

    return items


def _extract_core_subject(topic: str) -> str:
    """Extract core subject from verbose query (reuses openai_reddit logic)."""
    noise = ['best', 'top', 'how to', 'tips for', 'practices', 'features',
             'killer', 'guide', 'tutorial', 'recommendations', 'advice',
             'prompting', 'using', 'for', 'with', 'the', 'of', 'in', 'on',
             'what', 'are', 'people', 'saying', 'about']
    words = topic.lower().split()
    result = [w for w in words if w not in noise]
    return ' '.join(result[:4]) or topic  # Keep max 4 words


def _build_query_variants(core: str, full_topic: str) -> List[str]:
    """Build 2-3 Brave query variants for Reddit search.

    Compensates for Brave's weaker semantic search vs OpenAI web_search.
    """
    variants = []

    # Primary: core subject + site filter
    variants.append(f"{core} site:reddit.com")

    # Secondary: with "discussion" keyword
    variants.append(f"{core} discussion site:reddit.com")

    # Tertiary: if full topic differs from core, try full topic too
    if full_topic.lower().strip() != core.lower().strip():
        # Use a trimmed version of full topic
        trimmed = ' '.join(full_topic.split()[:6])  # Max 6 words
        variants.append(f"{trimmed} site:reddit.com")

    return variants[:3]  # Cap at 3 queries


def _search_brave(
    query: str,
    api_key: str,
    count: int,
    freshness: Optional[str],
) -> List[Dict[str, Any]]:
    """Execute a single Brave search query."""
    params = {
        "q": query,
        "result_filter": "web",  # No news for Reddit
        "count": count,
        "safesearch": "strict",
        "text_decorations": 0,
        "spellcheck": 0,
    }
    if freshness:
        params["freshness"] = freshness

    url = f"{ENDPOINT}?{urlencode(params)}"

    sys.stderr.write(f"[Brave Reddit] Searching: {query}\n")
    sys.stderr.flush()

    response = http.request(
        "GET", url,
        headers={"X-Subscription-Token": api_key},
        timeout=15,
    )

    # Extract web results only (news excluded to avoid non-reddit results)
    return response.get("web", {}).get("results", [])


def _normalize_reddit_url(url: str) -> Optional[str]:
    """Normalize Reddit URL for deduplication.

    Strips www/old prefix, trailing slashes, query params.
    Returns None if not a reddit.com URL.
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "reddit.com" not in host:
            return None
        # Normalize to reddit.com + path
        path = parsed.path.rstrip("/")
        return f"reddit.com{path}"
    except Exception:
        return None


def _extract_subreddit(url: str) -> str:
    """Extract subreddit name from Reddit URL."""
    match = re.search(r'/r/([^/]+)', url)
    return match.group(1) if match else ""


def _parse_brave_date(result: Dict) -> Optional[str]:
    """Parse date from Brave search result.

    Reuses the date parsing logic from brave_search module.
    """
    from .brave_search import _parse_brave_date as _parse_date
    return _parse_date(result.get("age"), result.get("page_age"))


def _is_reddit_thread(url: str) -> bool:
    """Check if URL is a Reddit thread (not just subreddit page, wiki, etc.)."""
    # Must contain /r/ and /comments/
    return "/r/" in url and "/comments/" in url


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    import html
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(text)
    return text


def _parse_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse Brave search results into Reddit item format.

    Returns items compatible with openai_reddit.parse_reddit_response() output.
    """
    items = []

    for result in results:
        url = result.get("url", "")
        if not url:
            continue

        # Filter: must be a Reddit thread URL
        if not _is_reddit_thread(url):
            continue

        # Filter: must be reddit.com domain
        try:
            host = urlparse(url).netloc.lower()
            if "reddit.com" not in host:
                continue
            # Skip non-user-facing Reddit domains
            if any(d in host for d in ["developers.reddit.com", "business.reddit.com"]):
                continue
        except Exception:
            continue

        title = _clean_html(str(result.get("title", "")).strip())
        snippet = _clean_html(str(result.get("description", "")).strip())
        date = _parse_brave_date(result)
        subreddit = _extract_subreddit(url)

        items.append({
            "id": f"R{len(items)+1}",
            "title": title[:200],
            "url": url,
            "subreddit": subreddit,
            "date": date,
            "why_relevant": snippet[:200] if snippet else "",
            "relevance": 0.65,  # Default for Brave results (no LLM ranking)
        })

    return items
