#!/usr/bin/env python3
"""The cell Lasso did not report: UNWATERMARKED vs UNWATERMARKED, different seeds.

Their churn = paired per-item disagreement between watermarked and unwatermarked
runs. Without a same-condition baseline that number has no scale. This measures
the baseline on a local model, at verdict level (the thing they actually score),
not token level.

Controls, run first, because the experiment means nothing without them:
  A. same seed twice  -> must be IDENTICAL, else `seed` is ignored and
     "different seeds" is really just resampling (still usable, but a different
     claim, and I need to know which one I am making).
  B. T=0.001          -> churn must be ~0, else the harness itself is noisy.
"""
import json, subprocess, sys, collections

MODEL = "qwen3:0.6b"
URL = "http://127.0.0.1:11434/api/generate"

ITEMS = [
    "The battery lasts all day and the screen is gorgeous.",
    "It broke after two weeks and support never replied.",
    "It arrived on time. The box was slightly dented.",
    "Honestly the best purchase I have made this year.",
    "Does what it says, nothing more, nothing less.",
    "Terrible value for the price, would not buy again.",
    "I was sceptical but it won me over completely.",
    "The manual is in another language entirely.",
    "Works, though the app crashes occasionally.",
    "Exceeded every expectation I had going in.",
    "Fine for the money but the build feels cheap.",
    "Absolute rubbish, it stopped charging on day three.",
    "Pretty good, a few rough edges in the software.",
    "I have no strong feelings about it either way.",
    "The design is lovely and it is a joy to use.",
    "It does the job but I would not recommend it.",
    "Shipping was fast, product is as described.",
    "Disappointing given how much it cost.",
    "Solid, dependable, no complaints at all.",
    "Mixed feelings — great hardware, poor software.",
]

PROMPT = ('Classify the sentiment of the review as exactly one word: '
          'positive, negative, or neutral.\nReview: "{}"\nAnswer with one word only.')


def gen(text, seed, temp):
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT.format(text), "stream": False,
        "think": False,
        "options": {"temperature": temp, "seed": seed, "num_predict": 8},
    })
    r = subprocess.run(["curl", "-sS", "--max-time", "120", URL, "-d", body],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout).get("response", "")
    except Exception:
        return ""


def verdict(raw):
    """Score the way a benchmark would: extract the discrete label."""
    t = raw.lower()
    for lab in ("positive", "negative", "neutral"):
        if lab in t:
            return lab
    return "UNPARSED"


def churn(temp, seed_a, seed_b, label):
    diffs, rows = 0, []
    for it in ITEMS:
        va, vb = verdict(gen(it, seed_a, temp)), verdict(gen(it, seed_b, temp))
        if va != vb:
            diffs += 1
        rows.append((va, vb, it[:38]))
    pct = 100.0 * diffs / len(ITEMS)
    print(f"  {label:<44} churn {diffs}/{len(ITEMS)} = {pct:.1f}%")
    for va, vb, t in rows:
        if va != vb:
            print(f"      {va:>9} -> {vb:<9}  {t}")
    return pct


print(f"  model {MODEL}, n={len(ITEMS)} items, verdict = extracted label\n")
print("  CONTROLS")
a = churn(0.001, 7, 7, "A: same seed (7 vs 7), T=0.001")
b = churn(1.0, 11, 11, "A': same seed (11 vs 11), T=1.0")
c = churn(0.001, 7, 99, "B: different seeds, T=0.001")
print("\n  THE MISSING CELL")
d = churn(1.0, 7, 99, "unwatermarked vs unwatermarked, T=1.0")
e = churn(0.7, 7, 99, "unwatermarked vs unwatermarked, T=0.7")

print("\n  READING")
if a > 0 or b > 0:
    print("   ⚠ same-seed runs disagree -> generation is not seed-reproducible here,")
    print("     so 'different seeds' below is plain resampling. Still the baseline")
    print("     I want, but it is a claim about resampling, not about seeds.")
else:
    print("   ✓ same seed reproduces exactly -> seed genuinely controls sampling.")
print(f"   verdict self-churn: T=0.001 {c:.1f}%   T=0.7 {e:.1f}%   T=1.0 {d:.1f}%")
print("   Lasso report, for comparison (phi-4, BFCL tool calls): 16.8% at T=1.0")
