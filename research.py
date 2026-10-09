"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic:
        return "topic"
    slug = re.sub(r'[^\w]+', '-', topic.lower()).strip('-')
    if not slug:
        return "topic"
    return slug[:60]


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"Please research the following topic and write a comprehensive report: {topic}"


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    subagent_calls = 0
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0
    
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for call in msg.tool_calls:
                name = call.get('name')
                if name:
                    tool_calls[name] += 1
                    if name == 'task':
                        subagent_calls += 1
                        
        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
            input_tokens += msg.usage_metadata.get('input_tokens', 0)
            output_tokens += msg.usage_metadata.get('output_tokens', 0)
            
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        }
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_content = files.get(REPORT_PATH)
    sources_content = files.get(SOURCES_PATH)
    
    if not report_content or not report_content.strip():
        raise RuntimeError("Report is missing or empty")
        
    if not sources_content:
        raise RuntimeError("sources.json is missing")
        
    try:
        if isinstance(sources_content, bytes):
            sources_content = sources_content.decode('utf-8')
        sources = json.loads(sources_content)
    except Exception:
        raise RuntimeError("sources.json is invalid JSON")
        
    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    source_families = sorted(list(set(s.get("source") for s in sources if s.get("source"))))
    
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": source_families
    }
    
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / f"{slug}.md").write_text(report_content if isinstance(report_content, str) else report_content.decode('utf-8'), encoding="utf-8")
    
    return reports_dir / f"{slug}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not topic:
        sys.stderr.write("Usage: python research.py <topic>\\n")
        return 2
        
    model = make_model()
    start = time.monotonic()
    
    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()
        })
        
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000}
            )
            elapsed = time.monotonic() - start
            model_name = os.environ.get("LAB_MODEL", "unknown")
            report_file = save_outputs(backend, topic, result["messages"], elapsed, model_name)
        except RuntimeError as e:
            sys.stderr.write(f"FAILED: {e}\\n")
            return 1
            
    print(f"Report saved to: {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
