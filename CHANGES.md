# Changes & rationale log

This tracks every change made to this repo on the `qwen3.5-ollama-local-eval`
branch, and *why*, since all of it was done for an initial local-model
(Ollama/Qwen3.5) pilot run that happens before the real research
implementation. Nothing here is upstream `siyuyuan/evoagent` behavior being
"improved" for its own sake — every edit exists because the pilot run hit it.

Baseline: `fc6d087` (`main`, = upstream `siyuyuan/evoagent`), unmodified.

Newest entries first. Add one entry per change, right after making it.

---

## 2026-10-06 — Calibration run analysis (no code change)

**What:** Ran the calibration command (direct prompting, 5 Codenames
instances, `max_tokens=4096`) agreed on the previous day. Reorganized each
run's log/result files into their own subfolder (commit `41dab7e`, done
directly by the user, not by me).

**Finding:** Raising the cap from 512 to 4096 (8x) did **not** fix the
truncation problem — it only moved the goalposts. 4 of 7 calls (57%) still
hit `done_reason=length` without ever writing "Final Answer:": 3 of 5
spymaster calls (idx 0, 2, 3) and 1 of the 2 guesser calls that got that far
(idx 1). Only 1 of 5 instances (idx 4) completed both sides cleanly (scored
1/2). The parse-guard added on 2026-10-05 did exactly what it was built for:
those 3 failed-spy instances were scored 0 and the guess call was skipped,
instead of leaking the raw reasoning as a hint.

**Implication (decision still open, not yet acted on):** this looks less
like "512 was too small" and more like the model doesn't reliably converge
to a final answer within *any* moderate budget on this prompt — simply
raising `max_tokens` again (e.g. to 8192) may just repeat the pattern.
Before picking Arm A/B/C's real budget, worth checking whether the open
`check_agent` argument-swap bug (still unfixed, see 2026-10-05 entry) or the
prompt wording itself is contributing, rather than treating this as a pure
token-budget problem.

**Files touched:** none by me. `41dab7e` moved (did not modify)
`spp/logs/calls_armB_smoke.jsonl` → `spp/logs/calls_armB_smoke/`,
`spp/logs/gpu_log_armB_smoke.csv` → same folder, and added the calibration
run's output under `spp/logs/calls_calib_direct_4096/` and
`spp/result/codenames_collaborative_qwen3.5:9b-q4_K_M_direct_calib_direct_4096/`.

---

## 2026-10-05 — Fix hint-leak via missing Final-Answer parse; bound retry loop

**Commit:** `4739a68`
**Files:** `spp/llm_evoagent_codenames.py`, `spp/util_func.py`

**Why:** The Arm B smoke test (previous entry) scored avg 0.30, and a review
caught that the score itself was invalid, not just noisy. In all 5 instances
the spymaster never wrote "Final Answer:" (it ran out of budget mid-essay),
so `answer.split("Final Answer:")[-1]` silently fell back to the *entire*
~2000-character raw reasoning text, which was then handed to the guesser as
the "hint" — and that raw text contains every target word verbatim. The
1.00 and 0.50 scores in that run were leaks, not successful plays.

Separately, `evaluator_construction`'s retry loop caught every exception
silently and retried up to 100,000 times with the error print commented
out — any real failure (bad request, Ollama down) would hang a run forever
with zero visible cause.

**What changed:**
- Added `extract_final_answer()` in `llm_evoagent_codenames.py`: returns
  `None` if `"Final Answer:"` is absent, instead of falling back to the raw
  text. Both the spy side and the guess side now score the instance 0 (and
  skip the guess call entirely on a spy-side failure) when this happens,
  and log `spy_parse_failed`/`guess_parse_failed` on the result record.
- `util_func.py`: the exception handler in `evaluator_construction` now
  prints the exception and gives up after 4 attempts total, instead of
  retrying silently up to 100,000 times.

**Known gap left open:** `spy_collaboration_func`/`guess_collaboration_func`
(the `evoagent`-method collaboration loop) have a separate, confirmed bug —
the quality-check sub-call is invoked as `message_construction(model_name,
prompt)` but the local `message_construction(prompt, model_name)` expects
the opposite argument order, so the check agent receives the literal string
`"qwen3.5:9b-q4_K_M"` as its entire prompt instead of the real check prompt.
Verified against the Arm B smoke test log: all 30 check calls had exactly
24 prompt tokens and an identical 153-token reply. This bug predates all
changes on this branch (confirmed present in `fc6d087`). It doesn't affect
`--method direct` (used by the calibration run), only `evoagent`/`refine`
methods — not yet fixed, needs fixing before Arm B is rerun.

---

## 2026-10-05 — Add live monitor.py; log prompt/answer/thinking text per call

**Commit:** `9865b28`
**Files:** `spp/util_func.py`, `spp/monitor.py` (new), `spp_requirements.lock.txt`

**Why:** During the Arm B smoke test there was no way to see what was
happening until the run finished — `evaluator_construction` was only
logging token counts (prompt/completion tokens, wall time, done_reason),
never the actual prompt/answer/thinking text. There was nothing to show a
live trace of, and the resulting `logs/calls.jsonl` was too large to read
by hand after the fact.

**What changed:**
- `util_func.py`: the per-call log record (`small`/Ollama path only) now
  also includes `messages` (prompt), `content` (answer), and `thinking`.
- `spp/monitor.py` (new): a `rich`-based live terminal dashboard. Tails the
  JSONL log and shows, in one pane: current arm/task/model, live GPU
  util/VRAM/power/temp, `ollama ps` offload status (to confirm 100% GPU and
  the right context size are actually in effect, not just requested),
  running totals (calls, tokens, tok/s, truncation rate), and a scrolling
  trace of recent calls with prompt/answer previews. This folds together
  three things that were previously separate terminal commands.
- `spp_requirements.lock.txt`: added `rich==15.0.0` and its transitive deps
  (`markdown-it-py`, `mdurl`, `pygments`).

**Known gap:** only the `data_type=small` (Ollama) path is logged;
azure/openai/gemini calls are not.

---

## 2026-10-05 — Wire Ollama/Qwen3.5 into spp's local-model path; Arm B smoke test

**Commit:** `a80038d`
**Files:** `spp/util_func.py`, `spp/llm_evoagent.py`, `spp/llm_evoagent_codenames.py`,
`spp_requirements.lock.txt` (new), `.gitignore` (new), smoke-test output under
`spp/logs/`, `spp/result/`

**Why:** The task was to run this repo's `spp` (NLP/Codenames) experiments
against a local Qwen3.5 9B model served by Ollama, as a pilot before the
real research work. The repo was never built for Ollama — it only talks to
Azure/OpenAI/Gemini, plus a `data_type=small` path originally pointed at a
FastChat/vLLM server on `localhost:8701`.

Two more things had to be fixed just to get the repo running at all,
independent of Ollama:
- Root `requirements.txt` pins `openai==1.14.1`, but `util_func.py` uses the
  pre-1.0 `openai.ChatCompletion`/`openai.api_base` API, which 1.x deleted
  outright (confirmed by actually hitting `APIRemovedInV1` when testing).
- `requirements.txt` never lists `langchain` or `google-generativeai`, even
  though `util_func.py` imports both unconditionally at module load time —
  so even the `small` path, which doesn't use either, still crashes on
  import without them.

Once pointed at Ollama, three more adjustments were needed for the actual
experiment design (three arms, same model+method pairing used by more than
one arm, 5-instance smoke tests):
- Ollama's **OpenAI-compatible** endpoint (`/v1/chat/completions`) silently
  ignores `options.num_ctx`/`options.num_gpu` and doesn't reliably support
  the `think` toggle — confirmed by `curl` testing both endpoints directly.
  Ollama's **native** API (`/api/chat`) honors all three.
- The results-JSONL filename was built as `progress_file.split('.')[0] +
  '.jsonl'`, which truncates at the *first* dot in the whole path. Harmless
  for dot-free model names (`gpt-4`, `llama-13b-chat`), but
  `qwen3.5:9b-q4_K_M` contains a dot, so every run's results file collapsed
  to `codenames_collaborative_qwen3.jsonl` — multiple arms sharing that
  model name would have silently overwritten each other's results.
- Arms A and C both use `method=direct` with the same `model_name`, so they
  would write to the same progress file (`{model_name}_{method}.txt`) and
  corrupt/merge each other's results with no error.

**What changed:**
- `util_func.py`, `data_type == 'small'` branch: rewritten to call Ollama's
  native `/api/chat` via `requests` instead of the legacy `openai` module.
  Reads `OLLAMA_BASE`, `OLLAMA_THINK`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_GPU`,
  `OLLAMA_MAX_TOKENS`, `OLLAMA_TEMPERATURE` from the environment (all with
  defaults matching the repo's original hardcoded values where one
  existed). Also writes one JSONL log record per call to `LOG_PATH`
  (default `logs/calls.jsonl`) with token counts, wall time, and
  `done_reason`.
- `llm_evoagent.py` / `llm_evoagent_codenames.py`: added `--run_tag`
  (suffix for progress/result filenames, so same-model+same-method arms
  don't collide) and `--limit` (process only the first N instances, for
  smoke tests); both now set `os.environ["INSTANCE_IDX"]` per loop
  iteration so log records can be joined back to a specific instance.
  Fixed the filename-truncation bug by switching to `os.path.splitext`.
- `spp_requirements.lock.txt` (new): a verified-working dependency set for
  Python 3.9 — `openai==0.28.1`, `langchain==0.1.9`,
  `langchain_core==0.1.32`, `google-generativeai==0.4.1`, plus the repo's
  existing pins. Root `requirements.txt` is left untouched (still wrong on
  its own) since fixing it isn't in scope — this lockfile is what actually
  gets installed for the local-model pilot.
- `.gitignore` (new): `__pycache__/`, `*.pyc`, `.venv*/`.
- Ran a 5-instance Codenames smoke test, Arm B config (EvoAgent, `--ind 3`,
  thinking off, `max_tokens=512`) — committed the output
  (`spp/logs/calls_armB_smoke.jsonl`, `spp/logs/gpu_log_armB_smoke.csv`,
  `spp/result/codenames_collaborative_..._armB_smoke.{jsonl,txt}`) for
  provenance. Result: avg score 0.30, ~50% of all calls (96% of
  answer-producing calls specifically) hit the 512-token cap — this is what
  led to the parse-guard fix two entries above.

**Also verified empirically (not code changes, but decisions this depended
on):** Ollama's `num_gpu` auto-detection was leaving ~20% of the model on
CPU by default on this 8GB-VRAM card regardless of context size; passing
`num_gpu: 999` forces full GPU offload. At `num_ctx=16384` with full
offload, only ~58MB of VRAM headroom remains free with Chrome/VS Code also
running — tight but workable for a smoke test, flagged as a risk for longer
runs.
