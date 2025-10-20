# AI-Guided Web Crawler

This project provides an AI-guided web crawler that ranks pages on a site by semantic relevance. It combines a conventional breadth-first crawl with a free to use transformer model (`all-MiniLM-L6-v2`) so the crawler can infer what to look for and prioritise pages that match the user goal. The application now launches directly into a polished graphical interface.

## Features

- Animated white-on-black Tkinter GUI with live logs, progress pulse, and sortable results table.
- Goal-aware filter that parses the user prompt, extracts the essential terms, and keeps only pages that match enough of them.
- Salary extraction and ranking so high-paying job listings surface first whenever the goal emphasises compensation.
- Embedding-based duplicate detection prevents reprocessing pages with near-identical content.
- Skips the landing page in the ranked results while still using it to establish the crawl goal.
- Respects `robots.txt` and stays on the starting domain by default.
- Uses `sentence-transformers` embeddings to infer an initial goal from the landing page when no goal is supplied.
- Ranks follow-up links by semantic similarity between their anchor text and the goal embedding.
- Summarises each visited page with the most relevant sentences, extracted keywords, and salary snippets when available.

## Installation

The dependencies are free to use. They rely on the `sentence-transformers` library (downloads `all-MiniLM-L6-v2`, about 90 MB) for embeddings and on standard web/GUI packages.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Graphical interface

```powershell
python crawler.py
```

Enter the start URL and optional goal, adjust the crawl limits, and press **Start crawl**. The animated banner and pulsing status dot show progress while logs and summaries stream into the lower panels. Each collected page displays keywords, representative sentences, salary snippets, and goal-alignment diagnostics. The interface keeps a white foundation with black accents so the results table and summaries stay readable.

## Example output

```
Fetching (1/25) https://example.com
...
Example Domain
https://example.com
Relevance: 0.842
  - This domain is for use in illustrative examples in documents.
  - You may use this domain in literature without prior coordination or asking for permission.

```

## Notes

- Always review the target website's terms of service before crawling.
- Increase the delay slider if you plan to crawl larger sites to avoid overloading servers.
- For offline usage, run the script once while connected to download the embedding model.
