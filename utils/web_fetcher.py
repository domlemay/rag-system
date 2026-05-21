"""Fetch a URL and extract the main readable text, stripping ads/nav/noise."""

from dataclasses import dataclass
from urllib.parse import urlparse

import trafilatura

# Domains known to be high-quality official documentation
_OFFICIAL_DOCS: frozenset = frozenset({
    "docs.python.org", "developer.mozilla.org", "docs.docker.com",
    "kubernetes.io", "docs.github.com", "docs.aws.amazon.com",
    "cloud.google.com", "learn.microsoft.com", "docs.sqlalchemy.org",
    "fastapi.tiangolo.com", "docs.djangoproject.com", "flask.palletsprojects.com",
    "docs.pytest.org", "numpy.org", "pandas.pydata.org", "pytorch.org",
    "reactjs.org", "vuejs.org", "svelte.dev", "nextjs.org",
    "redis.io", "postgresql.org", "mongodb.com", "graphql.org",
    "docs.rust-lang.org", "go.dev", "typescriptlang.org", "nodejs.org",
})

_BLOG_DOMAINS: frozenset = frozenset({
    "medium.com", "dev.to", "hashnode.dev", "substack.com",
    "blog.logrocket.com", "css-tricks.com", "smashingmagazine.com",
})


@dataclass
class FetchedContent:
    url: str
    domain: str
    raw_text: str
    url_source_hint: str   # "official_docs" | "blog" | "unknown"


def fetch(url: str) -> FetchedContent:
    """Download a URL and extract its main readable text.

    Raises ValueError if the page cannot be fetched or yields too little content.
    """
    raw_html = trafilatura.fetch_url(url)
    if not raw_html:
        raise ValueError(f"Could not fetch page: {url}")

    text = trafilatura.extract(
        raw_html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )
    if not text or len(text.strip()) < 150:
        raise ValueError(f"Could not extract meaningful content from: {url}")

    domain = urlparse(url).netloc.lstrip("www.")
    return FetchedContent(
        url=url,
        domain=domain,
        raw_text=text,
        url_source_hint=_classify_domain(domain),
    )


def _classify_domain(domain: str) -> str:
    if domain in _OFFICIAL_DOCS:
        return "official_docs"
    if domain in _BLOG_DOMAINS:
        return "blog"
    if domain.startswith("docs.") or "/docs/" in domain:
        return "official_docs"
    return "unknown"
