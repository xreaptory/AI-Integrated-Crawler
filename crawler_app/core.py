"""Core crawling logic and data structures."""

from __future__ import annotations

import heapq
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
from urllib import robotparser
from urllib.parse import urljoin, urlparse

USER_AGENT = "AICrawlerBot/1.0 (+https://example.com/bot)"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DUPLICATE_SIM_THRESHOLD = 0.92
MAX_SUMMARY_SENTENCES = 3
MAX_SUMMARY_WORDS = 90

STOP_WORDS = {
    "the",
    "and",
    "that",
    "with",
    "this",
    "from",
    "have",
    "will",
    "your",
    "about",
    "there",
    "their",
    "them",
    "http",
    "https",
    "www",
    "com",
}

GOAL_FILTER_STOPWORDS = {
    "find",
    "finding",
    "search",
    "searching",
    "look",
    "looking",
    "need",
    "needing",
    "want",
    "wanting",
    "seeking",
    "seek",
    "top",
    "best",
    "high",
    "higher",
    "highest",
    "pay",
    "paying",
    "salary",
    "salaries",
    "compensation",
    "gehalt",
    "lohn",
    "job",
    "jobs",
    "position",
    "positions",
    "role",
    "roles",
    "company",
    "companies",
    "firm",
    "firms",
    "team",
    "teams",
    "branch",
    "industry",
    "field",
    "sector",
    "area",
    "people",
    "candidate",
    "candidates",
    "work",
    "working",
    "experience",
    "experienced",
    "with",
    "without",
    "for",
    "from",
    "in",
    "and",
    "the",
    "a",
    "an",
}

LOCATION_SYNONYMS = {
    "austria": {"austria", "osterreich", "oesterreich"},
    "vienna": {"vienna", "wien"},
    "linz": {"linz"},
    "graz": {"graz"},
    "salzburg": {"salzburg"},
    "innsbruck": {"innsbruck"},
}

LOCATION_VARIANT_LOOKUP = {
    variant: synonyms for synonyms in LOCATION_SYNONYMS.values() for variant in synonyms
}

GOAL_PRIORITY_WORDS = {"top", "highest", "best", "leading", "lucrative"}
GOAL_SALARY_WORDS = {"pay", "paying", "salary", "salaries", "compensation", "gehalt", "lohn", "gehalter", "loehne"}

SALARY_SNIPPET_PATTERN = re.compile(
    r"(?:ab|ab\s+ca\.|von|zwischen)?\s*€\s*[\d\.\,]+(?:\s*(?:bis|–|-|to)\s*€?\s*[\d\.\,]+)?\s*(?:pro|per|je)?\s*(?:jahr|jaehrlich|jahrlich|year|monat|monatlich|month|woche|wochen|week|tag|tage|day|stunde|stunden|hour)",
    re.IGNORECASE,
)

SALARY_PERIOD_KEYWORDS = {
    "jahr": "year",
    "jaehr": "year",
    "jahrlich": "year",
    "jahres": "year",
    "annual": "year",
    "year": "year",
    "monat": "month",
    "monatlich": "month",
    "month": "month",
    "woche": "week",
    "wochen": "week",
    "week": "week",
    "tag": "day",
    "tage": "day",
    "day": "day",
    "stunde": "hour",
    "stunden": "hour",
    "hour": "hour",
}

SALARY_PERIOD_MULTIPLIERS = {
    "year": 1.0,
    "month": 12.0,
    "week": 52.0,
    "day": 260.0,
    "hour": 2080.0,
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_same_domain(url: str, root: str) -> bool:
    parsed_url = urlparse(url)
    parsed_root = urlparse(root)
    return parsed_url.netloc == parsed_root.netloc


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="")
    return clean.geturl()


def clean_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip()


def extract_text(soup: BeautifulSoup) -> str:
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    raw = " ".join(soup.stripped_strings)
    return clean_text(raw)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [clean_text(sentence) for sentence in parts if sentence.strip()]
    return sentences


def top_keywords(text: str, n: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    filtered = [token for token in tokens if token not in STOP_WORDS]
    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(n)]


def compose_summary(sentences: List[str], max_sentences: int = MAX_SUMMARY_SENTENCES, max_words: int = MAX_SUMMARY_WORDS) -> str:
    if not sentences:
        return ""
    summary: List[str] = []
    word_total = 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        summary.append(sentence)
        word_total += len(words)
        if len(summary) >= max_sentences or word_total >= max_words:
            break
    return " ".join(summary)


@dataclass(order=True)
class CrawlCandidate:
    priority: float
    url: str = field(compare=False)
    depth: int = field(compare=False)
    anchor: str = field(compare=False, default="")


@dataclass
class PageSummary:
    url: str
    title: str
    score: float
    summary_sentences: List[str]
    summary_text: str
    keywords: List[str]
    salary_text: str = ""
    salary_value: float = 0.0
    goal_match_terms: List[str] = field(default_factory=list)
    goal_missing_terms: List[str] = field(default_factory=list)
    goal_required_terms: List[str] = field(default_factory=list)
    goal_match_ratio: float = 1.0
    goal_match_passed: bool = True
    goal_must_terms: List[str] = field(default_factory=list)
    goal_missing_must: List[str] = field(default_factory=list)


@dataclass
class GoalFilterResult:
    accepted: bool
    matched_terms: List[str]
    missing_terms: List[str]
    required_terms: List[str]
    ratio: float
    must_terms: List[str]
    missing_must_terms: List[str]


@dataclass
class SalaryInfo:
    raw_text: str
    annual_eur: float
    period: str


class GoalFilter:
    """Heuristically filter pages so they align with the user's goal."""

    def __init__(self, goal_text: Optional[str]) -> None:
        self.goal_text = goal_text or ""
        self.required_terms: List[str] = []
        self._term_variants: Dict[str, Set[str]] = {}
        self._min_match = 0
        self._must_terms: List[str] = []
        self.prioritize_salary = False
        self.update_goal(goal_text)

    def update_goal(self, goal_text: Optional[str]) -> None:
        self.goal_text = goal_text or ""
        cleaned_goal = self.goal_text.strip()
        if not cleaned_goal:
            self.required_terms = []
            self._term_variants = {}
            self._min_match = 0
            self.prioritize_salary = False
            return
        goal_tokens = self._tokenize(cleaned_goal)
        tokens_for_priority = set(goal_tokens)
        location_terms = {variant for variants in LOCATION_SYNONYMS.values() for variant in variants}
        filtered_terms: List[str] = []
        for token in goal_tokens:
            if token in GOAL_FILTER_STOPWORDS and token not in location_terms:
                continue
            filtered_terms.append(token)
        if not filtered_terms:
            filtered_terms = list(goal_tokens)
        unique_terms = []
        seen: Set[str] = set()
        for term in filtered_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        self.required_terms = unique_terms
        self._term_variants = {}
        for term in self.required_terms:
            variants = set(self._expand_variants(term))
            self._term_variants[term] = variants
        location_token_set: Set[str] = set()
        for synonyms in LOCATION_SYNONYMS.values():
            for variant in synonyms:
                normalized = self._normalize_token(variant)
                if normalized:
                    location_token_set.add(normalized)
        total_terms = len(self.required_terms)
        self._min_match = 0 if total_terms == 0 else max(1, math.ceil(total_terms * 0.6))
        self._must_terms = []
        for term in self.required_terms:
            if term in location_token_set:
                continue
            if term in GOAL_PRIORITY_WORDS or term in GOAL_SALARY_WORDS:
                continue
            self._must_terms.append(term)
        self.prioritize_salary = bool(GOAL_PRIORITY_WORDS & tokens_for_priority) and bool(
            GOAL_SALARY_WORDS & tokens_for_priority
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        lowered = decomposed.lower()
        return "".join(char for char in lowered if not unicodedata.combining(char))

    def _normalize_token(self, token: str) -> str:
        base = self._normalize_text(token)
        return re.sub(r"[^a-z0-9\+\-]", "", base)

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.findall(r"[A-Za-z0-9\+\-]{3,}", text)
        normalized: Set[str] = set()
        for token in tokens:
            norm = self._normalize_token(token)
            if not norm:
                continue
            normalized.add(norm)
            if norm.endswith("s") and len(norm) > 4:
                normalized.add(norm[:-1])
        return normalized

    def _expand_variants(self, term: str) -> Set[str]:
        variants: Set[str] = {term}
        if term in LOCATION_SYNONYMS:
            variants.update(LOCATION_SYNONYMS[term])
        if term in LOCATION_VARIANT_LOOKUP:
            variants.update(LOCATION_VARIANT_LOOKUP[term])
        if term.endswith("s") and len(term) > 4:
            variants.add(term[:-1])
        return {self._normalize_token(variant) for variant in variants if variant}

    @property
    def min_match(self) -> int:
        return self._min_match

    def evaluate(self, text: str) -> GoalFilterResult:
        if not self._term_variants:
            return GoalFilterResult(True, [], [], [], 1.0, [], [])
        tokens = self._tokenize(text)
        matched: List[str] = []
        for term, variants in self._term_variants.items():
            if variants & tokens:
                matched.append(term)
        missing = [term for term in self._term_variants if term not in matched]
        total_terms = len(self._term_variants)
        ratio = float(len(matched)) / float(total_terms) if total_terms else 1.0
        missing_must = [term for term in self._must_terms if term not in matched]
        accepted = len(matched) >= self._min_match and not missing_must
        return GoalFilterResult(accepted, matched, missing, list(self._term_variants.keys()), ratio, list(self._must_terms), missing_must)


def _parse_salary_value(value: str) -> Optional[float]:
    cleaned = value.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_salary(text: str) -> Optional[SalaryInfo]:
    best: Optional[SalaryInfo] = None
    normalized_text = text.replace("\xa0", " ")
    ascii_text = "".join(
        char for char in unicodedata.normalize("NFKD", normalized_text) if not unicodedata.combining(char)
    )
    for match in SALARY_SNIPPET_PATTERN.finditer(ascii_text):
        snippet = match.group().strip()
        numbers = [value for value in re.findall(r"[\d\.\,]+", snippet)]
        parsed_values = [value for value in (_parse_salary_value(number) for number in numbers) if value]
        if not parsed_values:
            continue
        base_value = max(parsed_values)
        snippet_lower = snippet.lower()
        period = "year"
        for keyword, canonical in SALARY_PERIOD_KEYWORDS.items():
            if keyword in snippet_lower:
                period = canonical
                break
        multiplier = SALARY_PERIOD_MULTIPLIERS.get(period, 1.0)
        annual = base_value * multiplier
        if best is None or annual > best.annual_eur:
            best = SalaryInfo(raw_text=snippet, annual_eur=annual, period=period)
    return best


class RobotsCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            robots_url = urljoin(base, "/robots.txt")
            parser = robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
            except Exception:
                parser = None
            self._cache[base] = parser
        parser = self._cache[base]
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)


class AIDrivenCrawler:
    def __init__(
        self,
        root_url: str,
        query: Optional[str],
        max_pages: int,
        max_depth: int,
        delay: float,
        stay_in_domain: bool,
    ) -> None:
        self.root_url = normalize_url(root_url)
        self.query_text = query
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.stay_in_domain = stay_in_domain
        self.model = SentenceTransformer(MODEL_NAME)
        self.robots = RobotsCache()
        self.visited: Set[str] = set()
        self.results: List[PageSummary] = []
        self._query_embedding: Optional[np.ndarray] = None
        self._page_embeddings: List[np.ndarray] = []
        self._goal_filter = GoalFilter(query)
        self.goal_required_terms = list(self._goal_filter.required_terms)
        self.goal_min_match = self._goal_filter.min_match
        self.goal_prioritize_salary = self._goal_filter.prioritize_salary
        self.goal_text_resolved = ""

    def run(self) -> None:
        logging.info("Starting crawl at %s", self.root_url)
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        initial_text = self._fetch_text(session, self.root_url)
        if initial_text is None:
            logging.error("Unable to fetch the starting URL: %s", self.root_url)
            return

        sentences = split_sentences(initial_text)
        if not sentences:
            logging.error("No readable text found at the starting URL.")
            return

        if self.query_text:
            logging.info("Using user supplied goal: %s", self.query_text)
            query_text = self.query_text
        else:
            query_text = self._infer_goal(sentences)
            logging.info("Inferred goal from root page: %s", query_text)

        self._query_embedding = self.model.encode(query_text)
        self.goal_text_resolved = query_text
        self._goal_filter.update_goal(query_text)
        self.goal_required_terms = list(self._goal_filter.required_terms)
        self.goal_min_match = self._goal_filter.min_match
        self.goal_prioritize_salary = self._goal_filter.prioritize_salary
        if self.goal_required_terms:
            logging.info(
                "Goal heuristics expect at least %d term(s) from: %s",
                self.goal_min_match,
                ", ".join(self.goal_required_terms),
            )

        frontier: List[CrawlCandidate] = []
        heapq.heappush(frontier, CrawlCandidate(priority=-1.0, url=self.root_url, depth=0))
        initial_processed = False
        collected_pages = 0

        while frontier and collected_pages < self.max_pages:
            candidate = heapq.heappop(frontier)
            url = candidate.url
            if url in self.visited:
                continue
            if candidate.depth > self.max_depth:
                continue
            if self.stay_in_domain and not is_same_domain(url, self.root_url):
                continue
            if not self.robots.can_fetch(url):
                logging.debug("Blocked by robots.txt: %s", url)
                continue

            logging.info("Fetching (%d/%d) %s", len(self.visited) + 1, self.max_pages, url)
            time.sleep(self.delay)
            page_text, soup = self._fetch_page(session, url)
            if page_text is None or soup is None:
                continue

            analysis = self._summarise_page(url, soup, page_text)
            if analysis is None:
                self.visited.add(url)
                continue
            summary, page_embedding, filter_result = analysis
            heuristic_pass = filter_result.accepted
            is_seed_page = False
            if not initial_processed and url == self.root_url:
                initial_processed = True
                is_seed_page = True
            is_duplicate = self._is_duplicate_embedding(page_embedding)
            if is_duplicate:
                logging.info("Skipping near-duplicate page: %s", url)
            if not heuristic_pass:
                missing_terms = ", ".join(filter_result.missing_terms) if filter_result.missing_terms else "none"
                missing_must = (
                    ", ".join(filter_result.missing_must_terms) if filter_result.missing_must_terms else "none"
                )
                logging.debug(
                    "Goal heuristics flagged %s (missing terms: %s | missing must-have terms: %s)",
                    url,
                    missing_terms,
                    missing_must,
                )
            if not is_duplicate:
                self._page_embeddings.append(page_embedding)
            self.visited.add(url)
            summary.goal_match_passed = heuristic_pass

            include_in_results = not (is_seed_page or is_duplicate)
            if include_in_results and not heuristic_pass:
                logging.debug("Skipping %s due to unmet goal heuristics.", url)
                include_in_results = False

            if include_in_results:
                self.results.append(summary)
                collected_pages += 1

            for link, anchor_text in self._extract_links(soup, url):
                normalized = normalize_url(link)
                if normalized in self.visited:
                    continue
                if self.stay_in_domain and not is_same_domain(normalized, self.root_url):
                    continue
                priority = self._score_anchor(anchor_text)
                heapq.heappush(
                    frontier,
                    CrawlCandidate(priority=-priority, url=normalized, depth=candidate.depth + 1, anchor=anchor_text),
                )

    def _fetch_page(
        self, session: requests.Session, url: str
    ) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text" not in content_type:
                logging.debug("Skipped non-text content at %s", url)
                return None, None
            soup = BeautifulSoup(response.text, "html.parser")
            text = extract_text(soup)
            return text, soup
        except RequestException as error:
            logging.debug("Failed to fetch %s: %s", url, error)
            return None, None

    def _fetch_text(self, session: requests.Session, url: str) -> Optional[str]:
        text, _ = self._fetch_page(session, url)
        return text

    def _infer_goal(self, sentences: List[str]) -> str:
        embeddings = self.model.encode(sentences)
        centroid = np.mean(embeddings, axis=0)
        scored: List[Tuple[float, str]] = []
        for vector, sentence in zip(embeddings, sentences):
            score = cosine_similarity(vector, centroid)
            if sentence and score > 0:
                scored.append((score, sentence))
        scored.sort(reverse=True)
        top_sentences = [sentence for _, sentence in scored[:3]]
        keywords = top_keywords(" ".join(top_sentences))
        combined = " ".join(top_sentences)
        if keywords:
            combined += " Keywords: " + ", ".join(keywords)
        return combined

    def _score_anchor(self, anchor_text: str) -> float:
        if not anchor_text:
            return 0.1
        embedding = self.model.encode(anchor_text)
        return cosine_similarity(embedding, self._query_embedding)

    def _summarise_page(
        self, url: str, soup: BeautifulSoup, text: str
    ) -> Optional[Tuple[PageSummary, np.ndarray, GoalFilterResult]]:
        sentences = split_sentences(text)
        if not sentences:
            return None
        embeddings = self.model.encode(sentences)
        page_embedding = np.mean(embeddings, axis=0)
        page_score = cosine_similarity(page_embedding, self._query_embedding)

        scored_sentences = [
            (cosine_similarity(vector, self._query_embedding), sentence)
            for vector, sentence in zip(embeddings, sentences)
        ]
        scored_sentences.sort(reverse=True)
        summary_sentences = [sentence for _, sentence in scored_sentences[:MAX_SUMMARY_SENTENCES]]
        summary_text = compose_summary(summary_sentences)
        keywords = top_keywords(text)
        title_tag = soup.find("title")
        title = clean_text(title_tag.get_text()) if title_tag else url
        salary_info = extract_salary(text)
        filter_result = self._goal_filter.evaluate(text)
        return (
            PageSummary(
                url=url,
                title=title,
                score=page_score,
                summary_sentences=summary_sentences,
                summary_text=summary_text,
                keywords=keywords[:8],
                salary_text=salary_info.raw_text if salary_info else "",
                salary_value=salary_info.annual_eur if salary_info else 0.0,
                goal_match_terms=filter_result.matched_terms,
                goal_missing_terms=filter_result.missing_terms,
                goal_required_terms=filter_result.required_terms,
                goal_match_ratio=filter_result.ratio,
                goal_match_passed=filter_result.accepted,
                goal_must_terms=filter_result.must_terms,
                goal_missing_must=filter_result.missing_must_terms,
            ),
            page_embedding,
            filter_result,
        )

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Iterable[Tuple[str, str]]:
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not href:
                continue
            resolved = urljoin(base_url, href)
            parsed = urlparse(resolved)
            if parsed.scheme not in {"http", "https"}:
                continue
            anchor_text = clean_text(anchor.get_text())
            yield resolved, anchor_text

    def _is_duplicate_embedding(self, candidate: np.ndarray) -> bool:
        if not self._page_embeddings:
            return False
        for existing in self._page_embeddings:
            similarity = cosine_similarity(existing, candidate)
            if similarity >= DUPLICATE_SIM_THRESHOLD:
                return True
        return False

    def export(self) -> Dict[str, object]:
        return {
            "root": self.root_url,
            "query": self.query_text if self.query_text else "",
            "pages": [
                {
                    "url": page.url,
                    "title": page.title,
                    "score": page.score,
                    "summary": page.summary_sentences,
                    "summary_text": page.summary_text,
                    "keywords": page.keywords,
                    "salary_text": page.salary_text,
                    "salary_value": page.salary_value,
                    "goal_match_terms": page.goal_match_terms,
                    "goal_missing_terms": page.goal_missing_terms,
                    "goal_required_terms": page.goal_required_terms,
                    "goal_match_ratio": page.goal_match_ratio,
                    "goal_must_terms": page.goal_must_terms,
                    "goal_missing_must": page.goal_missing_must,
                }
                for page in sorted(self.results, key=lambda item: item.score, reverse=True)
            ],
        }


__all__ = [
    "AIDrivenCrawler",
    "CrawlCandidate",
    "GoalFilter",
    "GoalFilterResult",
    "PageSummary",
    "RobotsCache",
    "cosine_similarity",
    "extract_salary",
    "extract_text",
    "is_same_domain",
    "normalize_url",
    "split_sentences",
    "top_keywords",
    "USER_AGENT",
    "MODEL_NAME",
    "DUPLICATE_SIM_THRESHOLD",
]
