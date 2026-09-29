#!/usr/bin/env python3
"""What my published churn number left out: the model's own margin.

Two nights ago I published "20-45% of verdicts change identity on a re-roll at
T=1.0" and labelled the near-greedy 0% row a *control*. Reading Jeff -- a
0.8B classifier that returns a calibrated probability per option from one
forward pass and never samples at all -- made me ask the question I skipped:

  on the items that churn, what did the distribution look like?

If churn concentrates where the model is genuinely near 50/50, the sampler is
faithfully reporting uncertainty and my headline counts an honest coin-flip as
an error. If churn is spread across confident items too, the sampler is adding
noise unrelated to the model's belief. These are different claims and I
published neither.

PREDICTIONS, recorded before the run (2026-09-29):
  P1  churn concentrates on low-margin items; high-margin items churn ~0.
  P2  case-splitting ('Positive' and 'positive' are distinct tokens with
      near-equal logprobs) moves enough mass that at least one of 20 items has
      a different argmax under a naive token readout than under one that sums
      surface forms of the same label.

P2 matters because it is an argument for Jeff's design that its README does not
make: you train a readout not only for speed but because an untrained LM head
spreads one label across its spellings.
"""
import json, subprocess, collections, math, sys

MODEL = "qwen3:0.6b"
CHAT = "http://127.0.0.1:11434/api/chat"
LABELS = ("positive", "negative", "neutral")
N_SAMPLES = 16

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
    "Mixed feelings - great hardware, poor software.",
]

PROMPT = ('Classify the sentiment of the review as exactly one word: '
          'positive, negative, or neutral.\nReview: "{}"\nAnswer with one word only.')


def call(text, temp, seed=None, logprobs=False, npred=1):
    opts = {"temperature": temp, "num_predict": npred}
    if seed is not None:
        opts["seed"] = seed
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(text)}],
            "stream": False, "think": False, "options": opts}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 20
    r = subprocess.run(["curl", "-sS", "--max-time", "180", CHAT, "-d", json.dumps(body)],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


def to_label(tok):
    """Which label, if any, does this first token commit to?"""
    t = tok.strip().strip("*-_ ").lower()
    if not t:
        return None
    for lab in LABELS:
        if lab.startswith(t) and len(t) >= 3:
            return lab
    return None


def readout(text):
    """One forward pass. Return naive-argmax label, summed distribution, margins."""
    d = call(text, 0.0, seed=1234, logprobs=True)
    top = d["logprobs"][0]["top_logprobs"]
    # naive: the single highest-probability token, as an argmax readout would take it
    naive_tok = top[0]["token"]
    naive = to_label(naive_tok)
    # summed: pool every surface form of each label
    mass = collections.defaultdict(float)
    seen = 0.0
    for e in top:
        p = math.exp(e["logprob"])
        seen += p
        lab = to_label(e["token"])
        if lab:
            mass[lab] += p
    tot = sum(mass.values())
    dist = {k: v / tot for k, v in mass.items()} if tot else {}
    order = sorted(dist.items(), key=lambda kv: -kv[1])
    summed = order[0][0] if order else None
    margin = (order[0][1] - order[1][1]) if len(order) > 1 else (1.0 if order else 0.0)
    # margin a naive token-level readout would compute, over tokens not labels
    ps = [math.exp(e["logprob"]) for e in top]
    naive_margin = (ps[0] - ps[1]) / seen if len(ps) > 1 else 1.0
    return dict(naive=naive, naive_tok=naive_tok, summed=summed, dist=dist,
                margin=margin, naive_margin=naive_margin, label_mass=tot / seen,
                top20_mass=seen)


def verdict(raw):
    t = raw.lower()
    for lab in LABELS:
        if lab in t:
            return lab
    return "UNPARSED"


def churn_of(text):
    """Fraction of T=1.0 samples that disagree with the modal verdict."""
    vs = []
    for s in range(N_SAMPLES):
        d = call(text, 1.0, seed=1000 + s, npred=8)
        vs.append(verdict(d.get("message", {}).get("content", "")))
        print(".", end="", flush=True)
    mode, n = collections.Counter(vs).most_common(1)[0]
    return 1 - n / len(vs), mode, collections.Counter(vs)


def main():
    rows = []
    for i, text in enumerate(ITEMS):
        r = readout(text)
        ch, mode, counts = churn_of(text)
        r.update(item=text, churn=ch, mode=mode, counts=dict(counts))
        rows.append(r)
        print(f"  [{i+1:2}/20] margin {r['margin']:.3f} naive {r['naive']} "
              f"summed {r['summed']} churn {ch:.2f}")
    json.dump(rows, open("margin-results.json", "w"), indent=1)

    print("\n=== P2: does case-splitting flip an argmax? ===")
    flips = [r for r in rows if r["naive"] != r["summed"]]
    print(f"  naive-vs-summed disagreements: {len(flips)}/20")
    for r in flips:
        print(f"    {r['item'][:44]!r} naive={r['naive_tok']!r} -> {r['naive']} "
              f"| summed -> {r['summed']} {r['dist']}")
    mm = [r["margin"] - r["naive_margin"] for r in rows]
    print(f"  margin under summed readout is larger by "
          f"{sum(mm)/len(mm):+.3f} on average "
          f"(min {min(mm):+.3f}, max {max(mm):+.3f})")

    print("\n=== P1: does churn concentrate on low-margin items? ===")
    lo = [r for r in rows if r["margin"] < 0.5]
    hi = [r for r in rows if r["margin"] >= 0.5]
    for name, g in (("margin < 0.5", lo), ("margin >= 0.5", hi)):
        if g:
            print(f"  {name:14} n={len(g):2}  mean churn "
                  f"{sum(x['churn'] for x in g)/len(g):.3f}  "
                  f"max {max(x['churn'] for x in g):.2f}")
    # rank correlation, computed by hand (no scipy)
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    a, b = rank([r["margin"] for r in rows]), rank([r["churn"] for r in rows])
    n = len(rows)
    d2 = sum((a[i] - b[i]) ** 2 for i in range(n))
    print(f"  Spearman rho(margin, churn) = {1 - 6*d2/(n*(n*n-1)):+.3f}")

    print("\n=== how much of the distribution is even on-task? ===")
    lm = [r["label_mass"] for r in rows]
    print(f"  mass on the three labels, of the top-20 tokens: "
          f"mean {sum(lm)/len(lm):.3f}, min {min(lm):.3f}")
    print(f"  mean top-20 mass captured: "
          f"{sum(r['top20_mass'] for r in rows)/len(rows):.3f}")
    print(f"\n  overall churn across all items: "
          f"{sum(r['churn'] for r in rows)/len(rows):.3f}")


if __name__ == "__main__":
    main()
