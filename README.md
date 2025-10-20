# AI-Guided Web Crawler

This project provides an AI-guided web crawler that ranks pages on a site by semantic relevance. It combines a conventional breadth-first crawl with a free to use transformer model (`all-MiniLM-L6-v2`) so the crawler can infer what to look for and prioritise pages that match the user goal. The application now launches directly into a polished graphical interface.

## Features

- Animated white-on-black Tkinter GUI with live logs, progress pulse, and sortable results table.
- Goal-aware heuristics parse the user prompt, highlight the essential terms, and annotate how well each page overlaps with them.
- Embedding-based duplicate detection prevents reprocessing pages with near-identical content.
- Skips the landing page in the ranked results while still using it to establish the crawl goal.
- Respects `robots.txt` and stays on the starting domain by default.
- Uses `sentence-transformers` embeddings to infer an initial goal from the landing page when no goal is supplied.
- Ranks follow-up links by semantic similarity between their anchor text and the goal embedding.
- Summarises each visited page with relevant sentences and extracted keywords, with optional AI assistance.
- Optional OpenAI-powered summaries boost recall: provide an API key to get concise AI summaries and relevance overrides.

## Installation

The dependencies are free to use. They rely on the `sentence-transformers` library (downloads `all-MiniLM-L6-v2`, about 90 MB) for embeddings and on standard web/GUI packages.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

By default the app installs only free dependencies. If you want OpenAI-powered summaries, install the optional client after the main requirements:

```powershell
pip install openai
```

Without that extra step the crawler stays fully local and free.

## Graphical interface

```powershell
python crawler.py
```

Enter the start URL and optional goal, adjust the crawl limits, and press **Start crawl**. The animated banner and pulsing status dot show progress while logs and summaries stream into the lower panels. Each collected page displays keywords, representative sentences, and goal-alignment diagnostics. The interface keeps a white foundation with black accents so the results table and summaries stay readable.

### AI summaries (optional)

- Paste an OpenAI API key in the **OpenAI API key** field (kept only in memory) or set the `OPENAI_API_KEY` environment variable before launching the app.
- When a valid key is available the crawler asks the `gpt-4o-mini` model for JSON-formatted summaries and goal relevance judgements.
- The AI verdict can rescue promising pages the heuristic filter would reject or hide obvious mismatches.
- If the key is missing or the API call fails the crawler falls back to the built-in heuristic summaries automatically.

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
- The OpenAI summariser is optional: leave the key blank if you prefer to rely solely on the local heuristic filters.
