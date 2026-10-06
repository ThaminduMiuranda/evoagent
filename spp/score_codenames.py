"""Diagnostic scoring/analysis for a Codenames result file (and optionally its
matching call-log file), as produced by llm_evoagent_codenames.py.

Usage:
    python score_codenames.py --result result/.../foo.jsonl
    python score_codenames.py --result result/.../foo.jsonl --calls logs/.../foo_calls.jsonl
"""
import argparse
import json


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def marker_rate(records, field):
    # Older/upstream-mode runs never set spy_marker_present/guess_marker_present
    # (the field is absent, not False) -- treat those as "not recorded" rather
    # than as a 0% rate, which would be misleading.
    present = [r[field] for r in records if field in r and r[field] is not None]
    if not present:
        return "n/a (not recorded on this run)"
    return rate(sum(1 for v in present if v), len(present))


def hint_leak_rate(records):
    leaked = 0
    checked = 0
    for r in records:
        hint_word = r.get("hint_word")
        target_words = r.get("target_words")
        if hint_word is None or target_words is None:
            continue
        checked += 1
        hint_lower = hint_word.lower()
        if any(str(t).lower() in hint_lower for t in target_words):
            leaked += 1
    return rate(leaked, checked)


def score_result(path):
    records = read_jsonl(path)
    n = len(records)
    matched_total = sum(r.get("info", {}).get("matched_count", 0) for r in records)
    target_total = sum(r.get("info", {}).get("target_count", 0) for r in records)

    print(f"Result file: {path}")
    print(f"  instances: {n}")
    print(f"  pooled score: {rate(matched_total, target_total)}")
    print(f"  spymaster marker rate (Final Answer present): {marker_rate(records, 'spy_marker_present')}")
    print(f"  guesser marker rate (Final Answer present):   {marker_rate(records, 'guess_marker_present')}")
    print(f"  hint-leak rate (hint_word contains a target word): {hint_leak_rate(records)}")


def score_calls(path):
    records = read_jsonl(path)
    n = len(records)
    length_hits = sum(1 for r in records if r.get("done_reason") == "length")
    print(f"Call log: {path}")
    print(f"  calls: {n}")
    print(f"  truncated (done_reason == length): {rate(length_hits, n)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="Path to a result .jsonl from llm_evoagent_codenames.py")
    ap.add_argument("--calls", default=None, help="Optional path to the matching call-log .jsonl")
    args = ap.parse_args()

    score_result(args.result)
    if args.calls:
        print()
        score_calls(args.calls)
