"""YouTube search and transcript extraction via scrapetube + youtube-transcript-api.

Pure Python alternative to youtube_yt.py (yt-dlp backend). No external CLI needed.
Uses scrapetube for YouTube search and youtube-transcript-api for transcripts.

Follows the same interface as youtube_yt.py so the rest of the pipeline is unchanged.
"""

import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Depth configurations: how many videos to search / transcribe (same as youtube_yt.py)
DEPTH_CONFIG = {
    "quick": 10,
    "default": 20,
    "deep": 40,
}

TRANSCRIPT_LIMITS = {
    "quick": 3,
    "default": 5,
    "deep": 8,
}

# Max words to keep from each transcript
TRANSCRIPT_MAX_WORDS = 500


def _log(msg: str):
    """Log to stderr."""
    sys.stderr.write(f"[YouTube] {msg}\n")
    sys.stderr.flush()


def is_scrapetube_available() -> bool:
    """Check if scrapetube and youtube-transcript-api are importable."""
    try:
        import scrapetube  # noqa: F401
        from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_core_subject(topic: str) -> str:
    """Extract core subject from verbose query for YouTube search.

    Strips meta/research words to keep only the core product/concept name,
    same logic as youtube_yt.py (copied to avoid importing yt-dlp dependency).
    """
    text = topic.lower().strip()

    # Strip multi-word prefixes
    prefixes = [
        'what are the best', 'what is the best', 'what are the latest',
        'what are people saying about', 'what do people think about',
        'how do i use', 'how to use', 'how to',
        'what are', 'what is', 'tips for', 'best practices for',
    ]
    for p in prefixes:
        if text.startswith(p + ' '):
            text = text[len(p):].strip()

    # Strip individual noise words
    # NOTE: 'tips', 'tricks', 'tutorial', 'guide', 'review', 'reviews'
    # are intentionally KEPT — they're YouTube content types that improve search
    noise = {
        'best', 'top', 'good', 'great', 'awesome', 'killer',
        'latest', 'new', 'news', 'update', 'updates',
        'trending', 'hottest', 'popular', 'viral',
        'practices', 'features',
        'recommendations', 'advice',
        'prompt', 'prompts', 'prompting',
        'methods', 'strategies', 'approaches',
    }
    words = text.split()
    filtered = [w for w in words if w not in noise]

    result = ' '.join(filtered) if filtered else text
    return result.rstrip('?!.')


def _parse_view_count(text: str) -> int:
    """Parse view count text like '1,234 views', '1.2M views', 'No views'."""
    if not text:
        return 0
    text = text.lower().strip()
    if 'no view' in text:
        return 0

    # Remove "views" suffix and commas
    text = text.replace('views', '').replace('view', '').replace(',', '').strip()

    # Handle suffixes: K, M, B
    multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except (ValueError, TypeError):
                return 0

    try:
        return int(float(text))
    except (ValueError, TypeError):
        return 0


def _parse_relative_date(text: str) -> Optional[str]:
    """Parse relative date text like '3 weeks ago', '2 months ago' to YYYY-MM-DD."""
    if not text:
        return None
    text = text.lower().strip()

    # Match patterns like "3 weeks ago", "1 year ago", "2 days ago"
    match = re.match(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', text)
    if not match:
        # Handle "Streamed X ago" prefix
        match = re.match(r'streamed\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', text)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    now = datetime.now()
    if unit in ('second', 'minute', 'hour'):
        # Very recent — use today's date
        result = now
    elif unit == 'day':
        result = now - timedelta(days=amount)
    elif unit == 'week':
        result = now - timedelta(weeks=amount)
    elif unit == 'month':
        result = now - timedelta(days=amount * 30)
    elif unit == 'year':
        result = now - timedelta(days=amount * 365)
    else:
        return None

    return result.strftime('%Y-%m-%d')


def _safe_title(video: dict) -> str:
    """Extract title from scrapetube video dict."""
    title = video.get('title', {})
    if isinstance(title, dict):
        runs = title.get('runs', [])
        if runs and isinstance(runs, list):
            return runs[0].get('text', '')
        return title.get('simpleText', '')
    if isinstance(title, str):
        return title
    return ''


def _safe_channel(video: dict) -> str:
    """Extract channel name from scrapetube video dict."""
    owner = video.get('ownerText', {})
    if isinstance(owner, dict):
        runs = owner.get('runs', [])
        if runs and isinstance(runs, list):
            return runs[0].get('text', '')
    # Fallback: longBylineText
    long_byline = video.get('longBylineText', {})
    if isinstance(long_byline, dict):
        runs = long_byline.get('runs', [])
        if runs and isinstance(runs, list):
            return runs[0].get('text', '')
    return ''


def search_youtube(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> Dict[str, Any]:
    """Search YouTube via scrapetube. No API key needed.

    Args:
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'

    Returns:
        Dict with 'items' list of video metadata dicts.
    """
    try:
        import scrapetube
    except ImportError:
        return {"items": [], "error": "scrapetube not installed"}

    count = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    core_topic = _extract_core_subject(topic)

    _log(f"Searching YouTube for '{core_topic}' (since {from_date}, count={count})")

    try:
        videos = list(scrapetube.get_search(query=core_topic, limit=count, sort_by="relevance"))
    except Exception as e:
        _log(f"scrapetube search error: {e}")
        return {"items": [], "error": f"scrapetube error: {e}"}

    items = []
    for video in videos:
        video_id = video.get('videoId', '')
        if not video_id:
            continue

        title = _safe_title(video)
        channel_name = _safe_channel(video)

        # Parse view count
        view_text = ''
        vct = video.get('viewCountText', {})
        if isinstance(vct, dict):
            view_text = vct.get('simpleText', '') or vct.get('runs', [{}])[0].get('text', '')
        elif isinstance(vct, str):
            view_text = vct
        view_count = _parse_view_count(view_text)

        # Parse published date
        pub_text = ''
        ptt = video.get('publishedTimeText', {})
        if isinstance(ptt, dict):
            pub_text = ptt.get('simpleText', '')
        elif isinstance(ptt, str):
            pub_text = ptt
        date_str = _parse_relative_date(pub_text)

        items.append({
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel_name": channel_name,
            "date": date_str,
            "engagement": {
                "views": view_count,
                "likes": 0,  # Not available from search results
                "comments": 0,  # Not available from search results
            },
            "duration": video.get('lengthText', {}).get('simpleText', '') if isinstance(video.get('lengthText'), dict) else '',
            "relevance": 0.7,  # Default; no LLM relevance scoring for YouTube
            "why_relevant": f"YouTube video about {core_topic}",
        })

    # Soft date filter: prefer recent items but fall back to all if too few
    recent = [i for i in items if i["date"] and i["date"] >= from_date]
    if len(recent) >= 3:
        items = recent
        _log(f"Found {len(items)} videos within date range")
    else:
        _log(f"Found {len(items)} videos ({len(recent)} within date range, keeping all)")

    # Sort by views descending
    items.sort(key=lambda x: x["engagement"]["views"], reverse=True)

    return {"items": items}


def fetch_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript for a single video via youtube-transcript-api.

    Tries English first, falls back to any available language.

    Returns:
        Cleaned plaintext transcript, truncated to TRANSCRIPT_MAX_WORDS,
        or None if no transcript available.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except ImportError:
        return None

    api = YouTubeTranscriptApi()

    try:
        # Try English first
        result = api.fetch(video_id, languages=['en'])
    except (NoTranscriptFound,):
        # Fall back to any available language
        try:
            transcript_list = api.list(video_id)
            # Get the first available transcript
            for transcript in transcript_list:
                try:
                    result = transcript.fetch()
                    break
                except Exception:
                    continue
            else:
                return None
        except Exception:
            return None
    except (TranscriptsDisabled, VideoUnavailable):
        return None
    except Exception as e:
        _log(f"Transcript error for {video_id}: {e}")
        return None

    # Join all text segments into plaintext
    text_parts = [snippet.text for snippet in result]
    transcript = ' '.join(text_parts)

    # Clean up whitespace
    transcript = re.sub(r'\s+', ' ', transcript).strip()

    # Truncate to max words
    words = transcript.split()
    if len(words) > TRANSCRIPT_MAX_WORDS:
        transcript = ' '.join(words[:TRANSCRIPT_MAX_WORDS]) + '...'

    return transcript if transcript else None


def fetch_transcripts_parallel(
    video_ids: List[str],
    max_workers: int = 1,
) -> Dict[str, Optional[str]]:
    """Fetch transcripts serially (max_workers=1 to avoid rate limiting).

    Adds 1s delay between requests to avoid rate limiting.

    Args:
        video_ids: List of YouTube video IDs
        max_workers: Max parallel fetches (default 1, serial)

    Returns:
        Dict mapping video_id to transcript text (or None).
    """
    if not video_ids:
        return {}

    _log(f"Fetching transcripts for {len(video_ids)} videos")

    results = {}
    for i, vid in enumerate(video_ids):
        try:
            results[vid] = fetch_transcript(vid)
        except Exception:
            results[vid] = None

        # Rate limiting: 1s delay between requests
        if i < len(video_ids) - 1:
            time.sleep(1)

    got = sum(1 for v in results.values() if v)
    _log(f"Got transcripts for {got}/{len(video_ids)} videos")
    return results


def search_and_transcribe(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> Dict[str, Any]:
    """Full YouTube search: find videos via scrapetube, then fetch transcripts for top N.

    Args:
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'

    Returns:
        Dict with 'items' list. Each item has a 'transcript_snippet' field.
    """
    # Step 1: Search
    search_result = search_youtube(topic, from_date, to_date, depth)
    items = search_result.get("items", [])

    if not items:
        return search_result

    # Step 2: Fetch transcripts for top N by views
    transcript_limit = TRANSCRIPT_LIMITS.get(depth, TRANSCRIPT_LIMITS["default"])
    top_ids = [item["video_id"] for item in items[:transcript_limit]]
    transcripts = fetch_transcripts_parallel(top_ids)

    # Step 3: Attach transcripts to items
    for item in items:
        vid = item["video_id"]
        transcript = transcripts.get(vid)
        item["transcript_snippet"] = transcript or ""

    return {"items": items}


def parse_youtube_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse YouTube search response to normalized format.

    Returns:
        List of item dicts ready for normalization.
    """
    return response.get("items", [])
