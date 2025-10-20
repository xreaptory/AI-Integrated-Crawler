#!
# /usr/bin/env python3
"""AI-assisted web crawler that prioritizes pages based on semantic relevance."""

from __future__ import annotations

import heapq
import logging
import math
import queue
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import webbrowser

import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
from urllib import robotparser
from urllib.parse import urljoin, urlparse
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

USER_AGENT = "AICrawlerBot/1.0 (+https://example.com/bot)"
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_DEPTH = 2
DEFAULT_DELAY = 1.0
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DUPLICATE_SIM_THRESHOLD = 0.92

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

BG_COLOR = "#F7F7F7"
FG_COLOR = "#0D0D0D"
ACCENT_COLOR = "#101820"
ACCENT_LIGHT = "#1E2A38"
HIGHLIGHT_COLOR = "#FFFFFF"


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
    keywords: List[str]
    ai_summary: str
    salary_text: str = ""
    salary_value: float = 0.0
    goal_match_terms: List[str] = field(default_factory=list)
    goal_missing_terms: List[str] = field(default_factory=list)
    goal_required_terms: List[str] = field(default_factory=list)
    goal_match_ratio: float = 1.0
    goal_match_passed: bool = True
    goal_must_terms: List[str] = field(default_factory=list)
    goal_missing_must: List[str] = field(default_factory=list)


class QueueHandler(logging.Handler):
    """Route log messages into a thread-safe queue for the GUI."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            self.handleError(record)
            return
        self._queue.put(message)


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
        self._goal_filter.update_goal(query_text)
        self.goal_required_terms = list(self._goal_filter.required_terms)
        self.goal_min_match = self._goal_filter.min_match
        self.goal_prioritize_salary = self._goal_filter.prioritize_salary
        if self.goal_required_terms:
            logging.info(
                "Goal filter requires at least %d term(s) from: %s",
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
            include_in_results = True
            if not initial_processed and url == self.root_url:
                initial_processed = True
                include_in_results = False
            if self._is_duplicate_embedding(page_embedding):
                logging.info("Skipping near-duplicate page: %s", url)
                include_in_results = False
            if not filter_result.accepted:
                missing_terms = ", ".join(filter_result.missing_terms) if filter_result.missing_terms else "none"
                missing_must = (
                    ", ".join(filter_result.missing_must_terms) if filter_result.missing_must_terms else "none"
                )
                logging.info(
                    "Filtered out %s (missing terms: %s | missing must-have terms: %s)",
                    url,
                    missing_terms,
                    missing_must,
                )
                include_in_results = False
            self._page_embeddings.append(page_embedding)
            self.visited.add(url)
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
        summary_sentences = [sentence for _, sentence in scored_sentences[:3]]
        keywords = top_keywords(text)
        title_tag = soup.find("title")
        title = clean_text(title_tag.get_text()) if title_tag else url
        ai_summary = ""
        salary_info = extract_salary(text)
        filter_result = self._goal_filter.evaluate(text)
        return (
            PageSummary(
                url=url,
                title=title,
                score=page_score,
                summary_sentences=summary_sentences,
                keywords=keywords[:8],
                ai_summary=ai_summary,
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
                    "keywords": page.keywords,
                    "ai_summary": page.ai_summary,
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


class CrawlerGUI:
    """Tkinter interface that runs the crawler in a background thread."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Web Crawler")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)
        self.root.configure(bg=BG_COLOR)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Accent.TButton", background=ACCENT_COLOR, foreground=HIGHLIGHT_COLOR, padding=10, font=("Segoe UI", 11, "bold"))
        self.style.map(
            "Accent.TButton",
            background=[("pressed", ACCENT_LIGHT), ("active", ACCENT_LIGHT)],
            foreground=[("disabled", "#A0A0A0")],
        )
        self.style.configure("Accent.Horizontal.TProgressbar", troughcolor=HIGHLIGHT_COLOR, background=ACCENT_COLOR, lightcolor=ACCENT_COLOR, darkcolor=ACCENT_LIGHT, bordercolor=HIGHLIGHT_COLOR)
        self.style.configure("Treeview", background=HIGHLIGHT_COLOR, fieldbackground=HIGHLIGHT_COLOR, foreground=FG_COLOR, bordercolor=ACCENT_COLOR, rowheight=26)
        self.style.map("Treeview", background=[("selected", "#E6EBF5")], foreground=[("selected", ACCENT_COLOR)])

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.url_var = tk.StringVar()
        self.goal_var = tk.StringVar()
        self.max_pages_var = tk.IntVar(value=DEFAULT_MAX_PAGES)
        self.max_depth_var = tk.IntVar(value=DEFAULT_MAX_DEPTH)
        self.delay_var = tk.DoubleVar(value=DEFAULT_DELAY)
        self.allow_offsite_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Idle.")

        self.crawl_thread: Optional[threading.Thread] = None
        self._latest_results: List[PageSummary] = []
        self._displayed_results: List[PageSummary] = []
        self._is_crawling = False
        self._shine_offset = -220
        self._status_toggle = False
        self._progress_visible = False
        self.summary_link_var = tk.StringVar(value="No page selected")
        self._current_selection_url = ""
        self._goal_required_terms = []
        self._goal_min_match = 0
        self._goal_prioritize_salary = False

        self._build_layout()
        self._poll_log_queue()
        self._animate_banner()
        self._pulse_status()

    def _build_layout(self) -> None:
        self.banner = tk.Canvas(self.root, height=120, bg=BG_COLOR, highlightthickness=0, bd=0)
        self.banner.pack(fill=tk.X, padx=0, pady=(18, 6))
        self._banner_base = self.banner.create_rectangle(40, 18, self.root.winfo_width() - 40, 110, fill=ACCENT_COLOR, outline="")
        self._banner_shine = self.banner.create_rectangle(-220, 18, -80, 110, fill=HIGHLIGHT_COLOR, outline="", stipple="gray25")
        self._banner_title = self.banner.create_text(70, 50, anchor="w", text="AI Web Crawler", font=("Segoe UI", 26, "bold"), fill=HIGHLIGHT_COLOR)
        self._banner_subtitle = self.banner.create_text(
            70,
            86,
            anchor="w",
            text="Discover relevant pages with AI-guided prioritisation and live insights.",
            font=("Segoe UI", 11),
            fill="#E6E6E6",
        )
        self.banner.bind("<Configure>", self._update_banner_layout)

        form_card = tk.Frame(self.root, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        form_card.pack(fill=tk.X, padx=24, pady=(0, 14))
        form = tk.Frame(form_card, bg=HIGHLIGHT_COLOR)
        form.pack(fill=tk.X, padx=20, pady=18)

        tk.Label(form, text="Crawl Configuration", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(form, text="Start URL", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, columnspan=4, sticky="w", pady=(16, 4))
        self.url_entry = tk.Entry(
            form,
            textvariable=self.url_var,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.url_entry.grid(row=2, column=0, columnspan=4, sticky="ew")

        tk.Label(form, text="Goal (optional)", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, columnspan=4, sticky="w", pady=(16, 4))
        self.goal_entry = tk.Entry(
            form,
            textvariable=self.goal_var,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.goal_entry.grid(row=4, column=0, columnspan=4, sticky="ew")

        controls = tk.Frame(form, bg=HIGHLIGHT_COLOR)
        controls.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        controls.grid_columnconfigure(6, weight=1)

        tk.Label(controls, text="Max pages", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.pages_spin = tk.Spinbox(
            controls,
            from_=1,
            to=500,
            textvariable=self.max_pages_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.pages_spin.grid(row=0, column=1, padx=(8, 24))

        tk.Label(controls, text="Max depth", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w")
        self.depth_spin = tk.Spinbox(
            controls,
            from_=1,
            to=10,
            textvariable=self.max_depth_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.depth_spin.grid(row=0, column=3, padx=(8, 24))

        tk.Label(controls, text="Delay (s)", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=4, sticky="w")
        self.delay_spin = tk.Spinbox(
            controls,
            from_=0.0,
            to=10.0,
            increment=0.1,
            textvariable=self.delay_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.delay_spin.grid(row=0, column=5, padx=(8, 24))

        self.allow_offsite_check = tk.Checkbutton(
            controls,
            text="Allow offsite links",
            variable=self.allow_offsite_var,
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            selectcolor=HIGHLIGHT_COLOR,
            activebackground=HIGHLIGHT_COLOR,
            activeforeground=FG_COLOR,
            font=("Segoe UI", 10),
            highlightthickness=0,
        )
        self.allow_offsite_check.grid(row=0, column=6, padx=(0, 24))

        actions = tk.Frame(form_card, bg=HIGHLIGHT_COLOR)
        actions.pack(fill=tk.X, padx=20, pady=(0, 18))

        self.start_button = tk.Button(
            actions,
            text="Start crawl",
            command=self.start_crawl,
            bg=ACCENT_COLOR,
            fg=HIGHLIGHT_COLOR,
            activebackground=ACCENT_LIGHT,
            activeforeground=HIGHLIGHT_COLOR,
            font=("Segoe UI", 11, "bold"),
            padx=24,
            pady=10,
            borderwidth=0,
            relief="flat",
            cursor="hand2",
            disabledforeground="#8A8A8A",
        )
        self.start_button.pack(side=tk.LEFT)
        self.start_button.bind("<Enter>", lambda _: self._highlight_start(True))
        self.start_button.bind("<Leave>", lambda _: self._highlight_start(False))

        self.progress = ttk.Progressbar(actions, mode="indeterminate", style="Accent.Horizontal.TProgressbar", length=180)

        status_frame = tk.Frame(actions, bg=HIGHLIGHT_COLOR)
        status_frame.pack(side=tk.RIGHT)
        self.status_canvas = tk.Canvas(status_frame, width=18, height=18, bg=HIGHLIGHT_COLOR, highlightthickness=0, bd=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self._status_dot = self.status_canvas.create_oval(4, 4, 14, 14, fill="#B5B5B5", outline="")
        tk.Label(status_frame, textvariable=self.status_var, bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10)).pack(side=tk.LEFT)

        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))

        results_card = tk.Frame(content, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        results_card.pack(fill=tk.BOTH, expand=True, pady=(0, 14))
        tk.Label(results_card, text="Collected Pages", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=20, pady=(16, 6))

        tree_container = tk.Frame(results_card, bg=HIGHLIGHT_COLOR)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=20)
        columns = ("title", "score", "salary", "url")
        self.results_tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="browse")
        self.results_tree.heading("title", text="Title")
        self.results_tree.heading("score", text="Relevance")
        self.results_tree.heading("salary", text="Salary")
        self.results_tree.heading("url", text="URL")
        self.results_tree.column("title", width=260, anchor="w")
        self.results_tree.column("score", width=100, anchor="center")
        self.results_tree.column("salary", width=120, anchor="center")
        self.results_tree.column("url", width=360, anchor="w")
        vsb = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        self.results_tree.tag_configure("odd", background="#F0F2F5")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_selected)
        self.results_tree.bind("<Double-1>", self._on_result_activated)
        self.results_tree.bind("<Return>", self._on_result_activated)

        tk.Label(results_card, text="Selected Page Summary", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(12, 4))
        link_row = tk.Frame(results_card, bg=HIGHLIGHT_COLOR)
        link_row.pack(fill=tk.X, padx=20, pady=(0, 6))
        self.summary_link = tk.Label(
            link_row,
            textvariable=self.summary_link_var,
            bg=HIGHLIGHT_COLOR,
            fg="#808080",
            font=("Segoe UI", 10, "underline"),
            cursor="arrow",
        )
        self.summary_link.pack(side=tk.LEFT)
        self.summary_link.bind("<Button-1>", self._open_selected_url)
        self.open_button = tk.Button(
            link_row,
            text="Open in browser ↗",
            command=self._open_selected_url,
            bg=ACCENT_COLOR,
            fg=HIGHLIGHT_COLOR,
            activebackground=ACCENT_LIGHT,
            activeforeground=HIGHLIGHT_COLOR,
            padx=12,
            pady=4,
            borderwidth=0,
            relief="flat",
            cursor="arrow",
            state="disabled",
        )
        self.open_button.pack(side=tk.RIGHT)
        self.summary_text = ScrolledText(
            results_card,
            height=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            wrap=tk.WORD,
        )
        self.summary_text.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 16))
        self.summary_text.configure(borderwidth=0, highlightthickness=1, highlightbackground=ACCENT_COLOR)
        self.summary_text.tag_configure("summary_title", font=("Segoe UI", 11, "bold"))
        self.summary_text.tag_configure("hyperlink", foreground=ACCENT_COLOR, underline=True)
        self.summary_text.tag_bind("hyperlink", "<Enter>", lambda _: self.summary_text.config(cursor="hand2"))
        self.summary_text.tag_bind("hyperlink", "<Leave>", lambda _: self.summary_text.config(cursor="arrow"))
        self.summary_text.tag_bind("hyperlink", "<ButtonRelease-1>", self._open_selected_url)
        self.summary_text.bind("<Key>", lambda event: "break")

        log_card = tk.Frame(content, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        log_card.pack(fill=tk.BOTH, expand=True)
        tk.Label(log_card, text="Activity Log", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(16, 6))
        self.log_text = ScrolledText(
            log_card,
            height=8,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.log_text.configure(state="disabled", borderwidth=0, highlightthickness=1, highlightbackground=ACCENT_COLOR)

        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=1)
        form.columnconfigure(3, weight=1)
        self._set_summary_text("Results will appear here after the crawl finishes.")
        self._set_summary_link(None)
        self.url_entry.focus_set()

    def _update_banner_layout(self, event: tk.Event) -> None:
        width = max(event.width, 200)
        self.banner.coords(self._banner_base, 40, 18, width - 40, 110)
        self.banner.coords(self._banner_shine, self._shine_offset, 18, self._shine_offset + 160, 110)

    def _animate_banner(self) -> None:
        width = max(self.banner.winfo_width(), 200)
        self._shine_offset += 8
        if self._shine_offset > width + 120:
            self._shine_offset = -220
        self.banner.coords(self._banner_shine, self._shine_offset, 18, self._shine_offset + 160, 110)
        self.root.after(50, self._animate_banner)

    def _pulse_status(self) -> None:
        if self._is_crawling:
            self._status_toggle = not self._status_toggle
            fill = ACCENT_COLOR if self._status_toggle else ACCENT_LIGHT
        else:
            self._status_toggle = False
            fill = "#B5B5B5"
        self.status_canvas.itemconfig(self._status_dot, fill=fill)
        self.root.after(320, self._pulse_status)

    def _highlight_start(self, hover: bool) -> None:
        if self._is_crawling:
            return
        self.start_button.configure(bg=ACCENT_LIGHT if hover else ACCENT_COLOR)

    def _poll_log_queue(self) -> None:
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_output(message)
        self.root.after(150, self._poll_log_queue)

    def _append_output(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_busy_state(self, running: bool) -> None:
        self._is_crawling = running
        if running:
            self.start_button.configure(state="disabled", text="Crawling...", bg=ACCENT_LIGHT)
            if not self._progress_visible:
                self.progress.pack(side=tk.LEFT, padx=(20, 0))
                self._progress_visible = True
            self.progress.start(12)
        else:
            self.start_button.configure(state="normal", text="Start crawl", bg=ACCENT_COLOR)
            self.progress.stop()
            if self._progress_visible:
                self.progress.pack_forget()
                self._progress_visible = False

    def _set_summary_text(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.config(cursor="arrow")

    def _set_summary_link(self, url: Optional[str]) -> None:
        if url:
            display = url if len(url) <= 70 else url[:67] + "..."
            self.summary_link_var.set(display)
            self.summary_link.configure(fg=ACCENT_COLOR, cursor="hand2")
            self._current_selection_url = url
            self.open_button.configure(state="normal", cursor="hand2", bg=ACCENT_COLOR, fg=HIGHLIGHT_COLOR)
        else:
            self.summary_link_var.set("No page selected")
            self.summary_link.configure(fg="#808080", cursor="arrow")
            self._current_selection_url = ""
            self.open_button.configure(state="disabled", cursor="arrow", bg="#D8D8D8", fg="#6F6F6F")

    def _open_url(self, url: str) -> None:
        if not url:
            return
        try:
            logging.info("Opening in browser: %s", url)
            webbrowser.open(url)
        except Exception as error:  # pragma: no cover - platform browser failure
            logging.warning("Unable to open URL %s: %s", url, error)

    def _open_selected_url(self, _event: Optional[tk.Event] = None) -> None:
        self._open_url(self._current_selection_url)

    def _on_result_activated(self, _event: tk.Event) -> str:
        selection = self.results_tree.selection()
        if not selection:
            return "break"
        index = int(selection[0])
        if 0 <= index < len(self._displayed_results):
            page = self._displayed_results[index]
            self._open_url(page.url)
        return "break"

    def _clear_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._displayed_results = []
        self._goal_required_terms = []
        self._goal_min_match = 0
        self._goal_prioritize_salary = False
        self._set_summary_text("Crawl running... results will show here.")
        self._set_summary_link(None)

    def start_crawl(self) -> None:
        if self.crawl_thread and self.crawl_thread.is_alive():
            messagebox.showinfo("Crawler", "A crawl is already in progress.")
            return

        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Missing URL", "Please enter a starting URL.")
            return

        goal = self.goal_var.get().strip() or None
        try:
            max_pages = max(1, int(self.max_pages_var.get()))
            max_depth = max(1, int(self.max_depth_var.get()))
            delay = max(0.0, float(self.delay_var.get()))
        except ValueError:
            messagebox.showerror("Invalid input", "Please check the crawl settings.")
            return

        allow_offsite = self.allow_offsite_var.get()

        self._clear_results()
        self.status_var.set("Crawling...")
        self._set_busy_state(True)
        self._append_output(f"Starting crawl at {url}")

        self.crawl_thread = threading.Thread(
            target=self._run_crawler,
            args=(url, goal, max_pages, max_depth, delay, allow_offsite),
            daemon=True,
        )
        self.crawl_thread.start()

    def _run_crawler(
        self,
        url: str,
        goal: Optional[str],
        max_pages: int,
        max_depth: int,
        delay: float,
        allow_offsite: bool,
    ) -> None:
        try:
            crawler = AIDrivenCrawler(
                root_url=url,
                query=goal,
                max_pages=max_pages,
                max_depth=max_depth,
                delay=delay,
                stay_in_domain=not allow_offsite,
            )
            crawler.run()
        except Exception as exc:  # pragma: no cover - surfaced via GUI
            logging.exception("Crawler failed: %s", exc)
            self.root.after(0, lambda: self._on_crawl_failed(exc))
            return

        self.root.after(0, lambda: self._on_crawl_finished(crawler))

    def _populate_results(self, results: List[PageSummary]) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._displayed_results = results
        self._set_summary_link(None)
        if not results:
            message = "No pages were collected. Adjust crawl limits and try again."
            if self._goal_required_terms:
                message += " You can also soften the goal wording to relax the filter."
            self._set_summary_text(message)
            return
        for index, page in enumerate(results):
            tag = "odd" if index % 2 else "even"
            self.results_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(page.title, f"{page.score:.3f}", page.salary_text or "–", page.url),
                tags=(tag,),
            )
        self.results_tree.tag_configure("even", background=HIGHLIGHT_COLOR)
        first = self.results_tree.get_children()
        if first:
            self.results_tree.selection_set(first[0])
            self._show_result_details(results[0])

    def _show_result_details(self, page: PageSummary) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, page.title + "\n", ("summary_title",))
        self.summary_text.insert(tk.END, page.url + "\n", ("hyperlink",))
        self.summary_text.insert(tk.END, "\nRelevance score: {0:.3f}\n".format(page.score))
        if page.keywords:
            self.summary_text.insert(tk.END, "Keywords: {0}\n".format(", ".join(page.keywords)))
        if page.summary_sentences:
            self.summary_text.insert(tk.END, "\nKey sentences:\n")
            for sentence in page.summary_sentences:
                self.summary_text.insert(tk.END, f"- {sentence}\n")
        if page.salary_text:
            approx_value = f"{page.salary_value:,.0f}" if page.salary_value else ""
            if approx_value:
                self.summary_text.insert(tk.END, f"\nSalary: {page.salary_text} (≈ €{approx_value} per year)\n")
            else:
                self.summary_text.insert(tk.END, f"\nSalary: {page.salary_text}\n")
        elif self._goal_prioritize_salary:
            self.summary_text.insert(tk.END, "\nSalary: Not detected on this page.\n")
        if page.goal_required_terms:
            status = "met" if page.goal_match_passed else "partial"
            needed = max(self._goal_min_match, 1)
            total = len(page.goal_required_terms)
            matched_terms = ", ".join(page.goal_match_terms) if page.goal_match_terms else "none"
            missing_terms = ", ".join(page.goal_missing_terms) if page.goal_missing_terms else "none"
            self.summary_text.insert(
                tk.END,
                "\nGoal alignment ({status}, needs {needed}/{total} terms):\n".format(
                    status=status,
                    needed=needed,
                    total=total,
                ),
            )
            self.summary_text.insert(tk.END, f"  matched: {matched_terms}\n")
            if missing_terms != "none":
                self.summary_text.insert(tk.END, f"  remaining: {missing_terms}\n")
            if page.goal_must_terms:
                must_display = ", ".join(page.goal_must_terms)
                missing_must = (
                    ", ".join(page.goal_missing_must)
                    if page.goal_missing_must
                    else "none"
                )
                self.summary_text.insert(tk.END, f"  must-have terms: {must_display}\n")
                if missing_must != "none":
                    self.summary_text.insert(tk.END, f"  missing must-have: {missing_must}\n")
        self.summary_text.config(cursor="arrow")
        self._set_summary_link(page.url)

    def _on_result_selected(self, event: tk.Event) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._displayed_results):
            self._show_result_details(self._displayed_results[index])

    def _on_crawl_finished(self, crawler: AIDrivenCrawler) -> None:
        self._set_busy_state(False)
        self._goal_required_terms = list(crawler.goal_required_terms)
        self._goal_min_match = crawler.goal_min_match
        self._goal_prioritize_salary = crawler.goal_prioritize_salary
        results = crawler.results
        if self._goal_prioritize_salary:
            sorted_results = sorted(results, key=lambda item: (item.salary_value, item.score), reverse=True)
        else:
            sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
        self._latest_results = sorted_results
        count = len(sorted_results)
        base_status = f"Finished. {count} page(s) collected."
        self.status_var.set(base_status)
        self._append_output(f"Finished crawl. {count} page(s) collected.")
        if self._goal_required_terms:
            terms_display = ", ".join(self._goal_required_terms)
            self._append_output(
                f"Goal filter enforced: needs {self._goal_min_match} of [{terms_display}]."
            )
            if self._goal_prioritize_salary:
                self._append_output("Ranking favours higher salary offers for this goal.")
        self._populate_results(sorted_results)

    def _on_crawl_failed(self, error: Exception) -> None:
        self._set_busy_state(False)
        self.status_var.set("Error while crawling.")
        messagebox.showerror("Crawler error", str(error))

    def on_close(self) -> None:
        if self.crawl_thread and self.crawl_thread.is_alive():
            if not messagebox.askyesno("Quit", "A crawl is still running. Quit anyway?"):
                return
        logging.getLogger().removeHandler(self.log_handler)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
def main() -> None:
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.INFO)
    app = CrawlerGUI()
    app.run()


if __name__ == "__main__":
    main()
