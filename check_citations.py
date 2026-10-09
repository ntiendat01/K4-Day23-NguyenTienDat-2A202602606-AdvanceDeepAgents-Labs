"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    import re
    problems = []
    if not sources:
        return ["no sources in sources.json"]
    
    urls = set()
    source_ns = {}
    for src in sources:
        n = src.get("n")
        url = src.get("url", "")
        if not isinstance(n, int):
            problems.append(f"source missing or invalid 'n': {src}")
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid url: {url}")
        if url in urls:
            problems.append(f"duplicate url in sources.json: {url}")
        urls.add(url)
        source_ns[n] = src
        
    parts = report_text.split("## References")
    if len(parts) < 2:
        problems.append("missing '## References' heading")
        return problems
        
    body = parts[0]
    references = parts[1]
    
    cited_matches = re.findall(r'\[(\d+)\]', body)
    cited = set(int(n) for n in cited_matches)
    
    for n in cited:
        if n not in source_ns:
            problems.append(f"[{n}] cited but missing from sources.json")
            
    for n in source_ns:
        if n not in cited:
            problems.append(f"source [{n}] never cited")
            
    ref_lines = [line.strip() for line in references.split('\n') if re.match(r'^\[\d+\]', line.strip())]
    ref_numbers = []
    
    for line in ref_lines:
        match = re.match(r'^\[(\d+)\]', line)
        if match:
            n = int(match.group(1))
            ref_numbers.append(n)
            
            if n not in source_ns:
                problems.append(f"reference line for [{n}] but not in sources.json")
                
            urls_in_line = re.findall(r'https?://[^\s\]\)]+', line)
            if len(urls_in_line) != 1:
                problems.append(f"reference line [{n}] has {len(urls_in_line)} urls, expected exactly 1 (e.g. bundled sources)")
            elif n in source_ns and urls_in_line[0] != source_ns[n].get("url"):
                problems.append(f"reference line [{n}] url mismatch: {urls_in_line[0]} != {source_ns[n].get('url')}")
                
    if len(ref_numbers) != len(set(ref_numbers)):
        problems.append("duplicate source numbers in references list")
        
    for n in source_ns:
        if n not in ref_numbers:
            problems.append(f"source [{n}] is missing a reference line")
            
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
