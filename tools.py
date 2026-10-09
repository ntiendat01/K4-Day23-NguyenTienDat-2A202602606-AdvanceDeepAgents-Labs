"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    import random
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            delay = e.retry_after if getattr(e, 'retry_after', None) is not None else min(base * (2 ** attempt) + random.uniform(0, 1), cap)
            time.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504):
                if attempt == attempts - 1:
                    raise
                retry_after = e.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = min(base * (2 ** attempt) + random.uniform(0, 1), cap)
                time.sleep(delay)
            else:
                raise
        except httpx.TransportError as e:
            if attempt == attempts - 1:
                raise
            delay = min(base * (2 ** attempt) + random.uniform(0, 1), cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
_last_arxiv_call = 0.0

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    import re
    global _last_arxiv_call
    terms = re.findall(r'\w+', query)
    if not terms:
        return "NO RESULTS"
    
    try:
        now = time.time()
        elapsed = now - _last_arxiv_call
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _last_arxiv_call = time.time()
        
        search_query = " AND ".join(f"all:{t}" for t in terms)
        max_results_clamped = max(1, min(max_results, 30))
        
        def _call():
            resp = httpx.get(
                ARXIV_URL, 
                params={"search_query": search_query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": max_results_clamped, "start": 0},
                timeout=15.0
            )
            resp.raise_for_status()
            return resp
        
        resp = with_retry(_call)
        
        root = xml.etree.ElementTree.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        if not entries:
            return "NO RESULTS"
            
        records = []
        for entry in entries:
            id_el = entry.find('atom:id', ns)
            pub_el = entry.find('atom:published', ns)
            title_el = entry.find('atom:title', ns)
            sum_el = entry.find('atom:summary', ns)
            
            if id_el is None or pub_el is None or title_el is None or sum_el is None:
                continue
                
            raw_id = id_el.text.split('/abs/')[-1]
            id_val = re.sub(r'v\d+$', '', raw_id)
            title = re.sub(r'\s+', ' ', title_el.text).strip()
            summary = re.sub(r'\s+', ' ', sum_el.text).strip()[:600]
            
            records.append({
                "id": id_val,
                "url": f"https://arxiv.org/abs/{id_val}",
                "published": pub_el.text[:10],
                "title": title,
                "summary": summary
            })
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        limit = max(1, min(limit, 100))
        params = {"limit": limit}
        if date:
            params["date"] = date
            
        def _call():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp
            
        resp = with_retry(_call)
        items = resp.json()
        
        records = []
        for item in items:
            paper = item.get("paper", {})
            if not paper.get("id"):
                continue
                
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            
            if keyword:
                kw = keyword.lower()
                if kw not in title.lower() and kw not in summary.lower():
                    continue
                    
            records.append({
                "id": paper["id"],
                "url": f"https://huggingface.co/papers/{paper['id']}",
                "published": paper.get("publishedAt", "")[:10],
                "title": title,
                "summary": summary,
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
            
        records.sort(key=lambda x: x["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        limit = max(1, min(limit, 50))
        def _call():
            resp = httpx.get(HF_SEARCH_URL, params={"q": query, "limit": limit}, timeout=15.0)
            resp.raise_for_status()
            return resp
            
        resp = with_retry(_call)
        items = resp.json()
        
        records = []
        for item in items:
            paper = item.get("paper", {})
            if not paper.get("id"):
                continue
                
            records.append({
                "id": paper["id"],
                "url": f"https://huggingface.co/papers/{paper['id']}",
                "published": paper.get("publishedAt", "")[:10],
                "title": paper.get("title", ""),
                "summary": paper.get("ai_summary") or paper.get("summary", ""),
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
            
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        if not objective:
            objective = f"find information about {query}"
            
        api_key = os.environ.get("EXA_API_KEY", "")
        url = EXA_URL
        if api_key:
            url = f"{EXA_URL}?exaApiKey={api_key}"
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "objective": objective,
                    "numResults": num_results
                }
            }
        }
        
        def _call():
            resp = httpx.post(
                url,
                json=payload,
                headers={"Accept": "application/json, text/event-stream"},
                timeout=30.0
            )
            resp.raise_for_status()
            text = resp.text
            for line in text.splitlines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise Exception(f"Exa error: {data['error']}")
                    if "result" in data:
                        meta = data["result"].get("_meta", {})
                        if meta.get("rateLimit") or "rate limit" in str(meta).lower():
                            raise RetryableError("Exa rate limit hit", retry_after=5)
                        
                        contents = []
                        for content in data["result"].get("content", []):
                            if content.get("type") == "text":
                                contents.append(content["text"])
                        
                        full_text = "\n".join(contents)
                        if "rate limit" in full_text.lower() and "dashboard.exa.ai" in full_text.lower():
                            raise RetryableError("Exa rate limit hit", retry_after=5)
                            
                        return full_text
            return "NO RESULTS"
            
        res = with_retry(_call, cap=60.0)
        return res
    except Exception as e:
        err_msg = str(e)
        if "EXA_API_KEY" in os.environ and os.environ["EXA_API_KEY"]:
            err_msg = err_msg.replace(os.environ["EXA_API_KEY"], "[REDACTED]")
        return f"ERROR: {type(e).__name__}: {err_msg}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        api_key = os.environ.get("EXA_API_KEY", "")
        exa_url = EXA_URL
        if api_key:
            exa_url = f"{EXA_URL}?exaApiKey={api_key}"
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_fetch_exa",
                "arguments": {
                    "urls": [url]
                }
            }
        }
        
        def _call():
            resp = httpx.post(
                exa_url,
                json=payload,
                headers={"Accept": "application/json, text/event-stream"},
                timeout=30.0
            )
            resp.raise_for_status()
            text = resp.text
            for line in text.splitlines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise Exception(f"Exa error: {data['error']}")
                    if "result" in data:
                        meta = data["result"].get("_meta", {})
                        if meta.get("rateLimit") or "rate limit" in str(meta).lower():
                            raise RetryableError("Exa rate limit hit", retry_after=5)
                        
                        contents = []
                        for content in data["result"].get("content", []):
                            if content.get("type") == "text":
                                contents.append(content["text"])
                                
                        full_text = "\n".join(contents)
                        if "rate limit" in full_text.lower() and "dashboard.exa.ai" in full_text.lower():
                            raise RetryableError("Exa rate limit hit", retry_after=5)
                            
                        return full_text[:12000]
            return "NO RESULTS"
            
        res = with_retry(_call, cap=60.0)
        return res
    except Exception as e:
        err_msg = str(e)
        if "EXA_API_KEY" in os.environ and os.environ["EXA_API_KEY"]:
            err_msg = err_msg.replace(os.environ["EXA_API_KEY"], "[REDACTED]")
        return f"ERROR: {type(e).__name__}: {err_msg}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
