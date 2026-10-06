"""Self-check: build_spy_prompt() (SPY_FORMAT=upstream) must be byte-identical
to the original upstream f-string (commit fc6d087), not just visually similar.
Run directly: python check_build_spy_prompt.py
"""
import os

os.environ.setdefault("task", "code")

from llm_evoagent_codenames import build_spy_prompt

n = 2
target_words = ["apple", "banana"]
word_list = ["apple", "banana", "cherry", "date"]

# Reconstructed verbatim from `git show fc6d087:spp/llm_evoagent_codenames.py`
# (the original inline prompt in __main__, before it was extracted into
# build_spy_prompt). Trailing space after the closing \" on the second
# line is intentional and must be preserved.
upstream = f'''Try to find a single word hint that can accurately represent and link the {n} given words: "{target_words}". The key is to select a hint that does not cause confusion with other words from the following word list: {word_list}.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a single word from the word list>\" 
Answer:
'''

current = build_spy_prompt(n, target_words, word_list)

assert current == upstream, (
    f"build_spy_prompt() output is not byte-identical to upstream.\n"
    f"upstream repr: {upstream!r}\ncurrent  repr: {current!r}"
)
print("OK: build_spy_prompt(SPY_FORMAT=upstream) is byte-identical to fc6d087")
