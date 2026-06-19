"""Extract readable text from URLs."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
import trafilatura

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_generic(url: str, session: requests.Session | None = None) -> dict[str, str]:
    session = session or _session()
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=False)
        if not downloaded:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            downloaded = response.text

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc
        text = _normalize_whitespace(text or "")
        warning = ""
        if not text:
            warning = (
                "Could not extract readable text from this page. "
                "Some sites block automated fetching — open the page, copy the text, and paste it below."
            )
        return {
            "title": title or "Web page",
            "text": text,
            "source_url": url,
            "warning": warning,
        }
    except requests.RequestException as exc:
        return {
            "title": "Fetch failed",
            "text": "",
            "source_url": url,
            "warning": f"Could not fetch URL: {exc}",
        }


def fetch_url(url: str) -> dict[str, str]:
    url = url.strip()
    if not url:
        raise ValueError("URL is required")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return fetch_generic(url)