"""Live terminal dashboard for spp's local-model (Ollama) runs.

Run this in a second terminal while llm_evoagent.py / llm_evoagent_codenames.py
is running. It tails the per-call JSONL log util_func.py writes (one line per
model call) and redraws a live view: running token/truncation totals, GPU/VRAM
and `ollama ps` offload status, and a scrolling trace of recent calls with
prompt/answer previews.

Usage:
    python monitor.py --log logs/calls.jsonl
    python monitor.py --log logs/calls.jsonl --from-start   # replay an existing log
"""
import argparse
import collections
import json
import os
import subprocess
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        ).decode().strip()
        used, total, util, power, temp = [x.strip() for x in out.split(",")]
        return {
            "used": float(used),
            "total": float(total),
            "util": float(util),
            "power": float(power),
            "temp": float(temp),
        }
    except Exception:
        return None


def ollama_ps_snapshot(model_name):
    if not model_name:
        return None
    try:
        out = subprocess.check_output(["ollama", "ps"], timeout=2).decode()
    except Exception:
        return None
    for line in out.splitlines()[1:]:
        if model_name in line:
            return " ".join(line.split())
    return None


def truncate(text, width=70):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def prompt_preview(messages):
    for m in messages or []:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def tail_jsonl(path, from_start):
    while not os.path.exists(path):
        time.sleep(0.5)
    f = open(path, "r", encoding="utf-8")
    if not from_start:
        f.seek(0, os.SEEK_END)
    buf = ""
    while True:
        chunk = f.read()
        if chunk:
            buf += chunk
            parts = buf.split("\n")
            buf = parts.pop()
            for line in parts:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
        else:
            time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/calls.jsonl")
    ap.add_argument("--model", default=None, help="Defaults to the model seen in the log")
    ap.add_argument("--from-start", action="store_true", help="Replay the whole log instead of tailing new lines")
    ap.add_argument("--trace-rows", type=int, default=14)
    ap.add_argument("--refresh", type=float, default=2.0, help="Redraws per second")
    args = ap.parse_args()

    console = Console()
    totals = collections.defaultdict(float)
    n_calls = 0
    length_hits = 0
    empty_hits = 0
    last_arm = last_task = None
    model_name = args.model
    recent = collections.deque(maxlen=args.trace_rows)
    start_time = time.time()

    def render():
        gpu = gpu_snapshot()
        elapsed = max(time.time() - start_time, 1e-6)

        gpu_str = (
            f"GPU {gpu['util']:.0f}%  VRAM {gpu['used']/1024:.1f}/{gpu['total']/1024:.1f} GB  "
            f"{gpu['power']:.0f}W  {gpu['temp']:.0f}°C"
            if gpu
            else "GPU: n/a"
        )
        ps_str = ollama_ps_snapshot(model_name) or "ollama ps: (model not loaded)"
        header = (
            f"[bold]arm[/]={last_arm or '-'}  [bold]task[/]={last_task or '-'}  "
            f"[bold]model[/]={model_name or '-'}\n{gpu_str}\n{ps_str}"
        )

        stats = Table(title="Running totals", expand=True)
        for col in ("calls", "prompt_tok", "completion_tok", "tok/s", "avg s/call", "truncated", "empty"):
            stats.add_column(col, justify="right")
        tok_s = totals["completion_tokens"] / elapsed
        avg_s = totals["wall_time"] / n_calls if n_calls else 0
        trunc_pct = (length_hits / n_calls * 100) if n_calls else 0
        stats.add_row(
            str(n_calls),
            str(int(totals["prompt_tokens"])),
            str(int(totals["completion_tokens"])),
            f"{tok_s:.1f}",
            f"{avg_s:.1f}s",
            f"{length_hits} ({trunc_pct:.0f}%)",
            str(empty_hits),
        )

        trace = Table(title=f"Recent calls (last {len(recent)})", expand=True)
        for col, w in (("#", None), ("inst", None), ("done", None), ("ptok", None), ("ctok", None), ("prompt", 45), ("answer", 45)):
            trace.add_column(col)
        for rec in recent:
            done = rec.get("done_reason")
            done_fmt = f"[red]{done}[/]" if done == "length" else f"[green]{done}[/]"
            trace.add_row(
                str(rec["_n"]),
                str(rec.get("instance_idx")),
                done_fmt,
                str(rec.get("prompt_tokens")),
                str(rec.get("completion_tokens_total")),
                truncate(rec["_prompt"], 45),
                truncate(rec["_answer"], 45),
            )

        out = Table.grid(expand=True)
        out.add_row(Panel(header, title="spp / Ollama live monitor"))
        out.add_row(stats)
        out.add_row(trace)
        return out

    with Live(render(), console=console, refresh_per_second=args.refresh, screen=False) as live:
        for rec in tail_jsonl(args.log, args.from_start):
            n_calls += 1
            last_arm = rec.get("arm") or last_arm
            last_task = rec.get("task") or last_task
            if model_name is None:
                model_name = rec.get("model")
            totals["prompt_tokens"] += rec.get("prompt_tokens") or 0
            totals["completion_tokens"] += rec.get("completion_tokens_total") or 0
            totals["wall_time"] += rec.get("wall_time_sec") or 0
            if rec.get("done_reason") == "length":
                length_hits += 1
            if rec.get("content_empty"):
                empty_hits += 1

            rec["_n"] = n_calls
            rec["_prompt"] = prompt_preview(rec.get("messages"))
            rec["_answer"] = rec.get("content") or rec.get("thinking") or ""
            recent.append(rec)

            live.update(render())


if __name__ == "__main__":
    main()
