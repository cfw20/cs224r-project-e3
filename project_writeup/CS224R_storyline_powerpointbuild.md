# CS224R Project — Storyline & Source-of-Truth

**Project:** *Unlocking Diversity in RL-Trained Reasoning Models*
**Authors:** Anna Grebenchtchikova & Chung Wong
**Course:** Stanford CS224R, Spring 2026
**Working title for the talk:** *Two Roads to Diversity: Input Noise Sharpens, Negative Gradients Explore*

> Mode: writing only (out of Modal credits) — no new experiments. Everything below is from runs already completed.

---

## 1. The one-sentence thesis

"Diversity" in RL-for-reasoning is **two knobs, not one.** Content-free **input noise** (random numeral-free trivia prepended to training prompts) **sharpens** the policy (entropy goes *down*); the **negative gradient** in e3 **explores** (entropy goes *up*). They are distinct, even opposing, interventions that happen to share the word "diversity."

The practical payoff: trivia buys **efficiency at no in-distribution accuracy cost**, and combined with e3 sits at the **sweet spot** (most token-efficient *and* closest to the base model). The **boundary condition** is long-horizon, out-of-distribution transfer (MATH → AIME), where the sharpening *breaks*.

---

## 2. Research question

Does **content-free input variation** (random trivia, training-only; eval always clean) improve a GRPO-trained Qwen3-1.7B reasoner — and how does it compare to:
- the **gradient-side** intervention (e3's negative gradient / exploration bonus), and
- the **content-rich** intervention (RLAD's learned abstractions)?

Framed against RLAD's own inconsistency: if content-rich hints help on some benchmarks and hurt on others, **how much of the effect is the *content* of a hint vs. merely *perturbing the input*?**

---

## 3. The 8-slide storyline (slides 0–7) — the part to preserve

**Slide 0. Title slide.**

**Slide 1 — "Compute is the new gold — and 'diversity' is how RL spends it"** (1/3 ↔ 2/3)
Left third: the stakes — RL-for-reasoning explores, exploration is expensive (a simple pictogram: $ / GPU → tokens → reasoning). Right two-thirds: your one-line thesis stated up front — diversity is two knobs, not one: input noise sharpens, the negative gradient explores. Sets the whole deck.

**Slide 2 — "Today's recipe for 'diversity' gives inconsistent returns"** (1/2 ↔ 1/2)
Left: the RLAD inconsistency, shown as a mini heat-strip or up/down arrows (helps base ↑, hurts DAPO ↓, varies by benchmark) — make the messiness visual. Right: the open question that falls out — how much is the content, vs. just perturbing the input? This is your motivation slide.

**Slide 3 — "A one-line trick matched accuracy at lower cost — so we stress-tested it"** (2/3 ↔ 1/3)
Left two-thirds: Chung's hook (Image 1 — the two accuracy curves sitting on top of each other) with a "same accuracy" callout. Right third: your experiment grid as a clean 3-block diagram (GRPO clean/trivia · e3 clean/trivia/mixed · MATH clean/trivia), with the "step-limited, mixture not more data" note as a footnote strip.

**Slide 4 — "Trivia keeps accuracy and reaches peak on roughly half the tokens"** (3/4 ↔ 1/4)
Three-quarters: Fig 2 (the efficiency frontier) — your hero figure. Right quarter: the takeaway stack — accuracy preserved · fewer truncations · ~21M vs ~40M tokens-to-peak, as three stacked blocks with a down-arrow on "tokens."

**Slide 5 — "On MATH the model gets dramatically leaner for a 2-point price"** (3/4 ↔ 1/4)
Three-quarters: Fig 3 MATH panel (727 vs 1325). Right quarter: the throughput-not-clock framing as a small block (same speed, fewer tokens → finished in budget; clean had to be stitched) and the forward hook → content-aware (Fibonacci) trivia could do even better.

*AIME addition — MATH: leaner, but fragile out-of-distribution.* On MATH the model becomes ~45% leaner in-distribution for ~2pp accuracy. But AIME transfer collapses 6× (0.146 → 0.023). Wall-clock note: MATH is the clean comparison — matched throughput; trivia's shorter generations finished in budget, clean exceeded the 24h cap and was stitched. Forward hook: content-aware (Fibonacci / domain) trivia as future work.

**Slide 6 — "The mechanism: trivia sharpens, e3 explores — opposite knobs"** (1/2 ↔ 1/2)
Left: Fig 4 (KL drift) — e3 stays closest to base, your trivia+mixed closest of all. Right: the conceptual payoff as a two-arrow diagram (input noise → entropy ↓ / sharpen; negative gradient → entropy ↑ / explore), plus the honest twist as a one-line callout (low drift ≠ guaranteed generalisation — MATH-trivia stayed close yet transferred worst). Entropy lives here as one sentence, not a chart.

*AIME addition — Mechanism: two opposing knobs + the twist.* KL-drift figure: all e3 runs stay far closer to base than either GRPO run; within e3, trivia & mixed are closest. AIME (GSM8K-trained) bars: differences are within error bars — the only clear lever is the e3 gradient. The two-knob diagram: input noise → entropy ↓ (sharpen); negative gradient → entropy ↑ (explore). The twist / counterexample: MATH-trivia drifts least from base yet transfers worst (AIME 0.023) — so low KL is necessary, not sufficient for generalization (a caveat to e3's "low-KL → generalize" intuition). Entropy is one sentence, not a hero figure.

**Slide 7 — "Together they're the sweet spot — and here's where it goes next"** (2/3 ↔ 1/3)
Left two-thirds: the synthesis — e3 + trivia = most token-efficient AND closest to base, shown as a small 2×2 or a converging-arrows visual. Right third: future work as an arrow timeline (domain trivia on MATH → more AIME conditions → scale to o4-mini, tying to RLAD's weak-to-strong).

---

## 4. Verified data (source of truth)

7 training runs, all **400 steps**, batch 64, rollout.n 8 → **25,600 samples each**. Step-limited (mixed = batch mixture, not extra compute).

**GSM8K validation accuracy — final | peak**

| Track | Run | Final | Peak |
|---|---|---|---|
| A | GRPO clean | .861 | .862 |
| B | GRPO trivia | .846 | .871 |
| C | e3 clean (partial e3) | .883 | .889 |
| D | e3 trivia (partial e3) | .883 | .883 |
| E | e3 mixed (2M partial e3) | .878 | .881 |

Trivia leads clean at steps 100–300 and peaks higher; e3 adds ~2pp.

**MATH validation accuracy — final | peak:** clean .724 | .733; trivia .686 | .711 (unstable).

**AIME 2025 pass@1** (n=8, 60 problems, SE ≈ 0.05):

| Track | pass@1 |
|---|---|
| A GRPO clean | .173 |
| B GRPO trivia | .154 |
| C e3 clean | .217 |
| D e3 trivia | .196 |
| E e3 mixed | .202 |
| MATH clean | .146 |
| MATH trivia | .023 |

GSM8K AIME differences are within noise; the e3 lift (A→C +.044) is the only clear effect. MATH-trivia is a real **6× collapse** (and had the lowest extract-fail, 18%).

**Tokens-to-90%-of-peak (millions, hardware-independent):** D 20.8 (best), C 29.6, E 30.4, A 39.3, B 42.1.

**Validation response length** (overall key, excluding `0_/1_length`):
- GSM8K final — A 587, B 537, C 630, D 668, E 665 → **small; curves cross** (trivia longer early, ~tied late — *not* a clean effect).
- MATH — **1325 (clean) vs 727 (trivia) ≈ 45% leaner** — the real leanness effect.
- (Note: an earlier 811/496 figure was a bug from 3 keys ending in `length/mean`.)

**KL drift to base — final | mean:** A .510|.190, B .454|.228, C .218|.147, D .163|.110, E .182|.111, MATH clean .288|.117, MATH trivia .105|.107.
- Robust claim: **all e3 ≪ both GRPO**; within e3, D & E closest to base.
- MATH-trivia has the **lowest KL yet worst OOD** — the counterexample.

**Truncation:** GSM8K 16% → 14%; MATH 36% → 29% (trivia lower).
**Trivia prompt overhead:** +7.3 tokens.
**Entropy:** GRPO/trivia collapse (.019 / .025); e3 high (.13–.18). Trivia *sharpens* (entropy down); e3's negative gradient *explores* (entropy up).
**Wall-clock:** trivia consistently faster; MATH is the clean matched-throughput case (~6.1–6.5k tok/s); clean MATH exceeded the 24h cap and was stitched. W&B doesn't log runtime cleanly → **lead efficiency with tokens-to-peak; lead wall-clock only with the MATH "finished in budget vs. stitched" line.**

---

## 5. RLAD Table 2 (for slide 2) — Qwen3-1.7B

Columns are three benchmarks × {w/o abs, w/abs (avg), w/abs (best)}.

| Approach | AIME w/o | AIME w/abs | AIME best | DSR w/o | DSR w/abs | DSR best | AMC w/o | AMC w/abs | AMC best |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 33.75 | 36.25 ▲ | 40.00 | 20.21 | 22.14 ▲ | 32.50 | 86.41 | 78.01 ▼ | 84.53 |
| + DAPO | 37.92 | 34.90 ▼ | 39.79 | 21.67 | 21.88 ▲ | 33.54 | 86.41 | 81.99 ▼ | 88.44 |
| + RLAD | 38.04 | 42.45 ▲ | 48.33 | 23.54 | 24.84 ▲ | 35.54 | 87.25 | 88.35 ▲ | 91.72 |

Arrows compare **w/o → w/abs (avg)**: ▲ = abstractions help, ▼ = abstractions hurt. The story: abstractions help the base on AIME/DSR but **hurt on AMC**; **hurt DAPO on AIME**; help RLAD everywhere. Source: Liu et al., *RLAD* (2025), Table 2.

---

## 6. Figures (what each shows)

- **Pareto (slide 4):** GSM8K held-out accuracy vs **cumulative tokens generated** (x-axis), rings at 90%-of-peak. Hero efficiency figure.
- **Response length (slide 5):** two panels — GSM8K (small, curves cross) and MATH (clear ~45% leaner). MATH-only version for slide 5.
- **KL drift (slide 6):** GSM8K and MATH panels; all e3 below both GRPO; MATH-trivia lowest.
- **Entropy phase (optional / one-line):** entropy vs accuracy; GRPO & trivia collapse, e3 stays high. *Use as a sentence, not a hero figure.*
- **AIME bars — GSM8K-trained (slide 6):** pass@1 with SE error bars; differences within noise; only e3 lift clear.
- **AIME bars — MATH (slide 5):** clean .146 vs trivia .023; the 6× collapse.
- **Chung clean-vs-mixed (slide 3):** GSM8K accuracy curves overlapping ("same accuracy, lower cost").

**Consistent colour encoding (cool = clean, warm = trivia):**
A GRPO clean = navy; B GRPO trivia = orange; C e3 clean = blue; D e3 trivia = red; E e3 mixed = mid-blue (dashed).

---

## 7. Framing rules / things to get right

- **Step-limited** everywhere: "mixed/2M" = 50/50 batch mixture, **not** more compute or data. Never imply otherwise.
- **AIME is secondary / "nice to have"**, always shown **with error bars** (SE ≈ 0.05, 60 problems). Do not over-claim GSM8K AIME differences — they are within noise.
- **Efficiency** = tokens-to-peak (hardware-independent). Wall-clock only via the MATH "finished in budget vs. stitched" line.
- **Compute-as-gold** is a **motivation/north-star**, not a result.
- **The twist** (low KL ≠ guaranteed transfer; MATH-trivia closest to base yet worst on AIME) is the intellectual payoff of slide 6 — keep it.
- **Entropy**: one sentence, not a figure.

## 8. Papers to cite (and how)

- **e3** (Setlur et al., 2025) — read deep; the gradient-side / exploration comparison and the "low-KL → extrapolation" intuition we complicate.
- **RLAD** (Liu et al., 2025) — read deep; Table 2 inconsistency is the slide-2 motivation; content-rich abstractions are the contrast to content-free trivia.
- **DAPO** (Yu et al., 2025) — one line (baseline in RLAD table).
- **Yue et al. (2025)** — one real sentence: sharpening vs. exploration ("does RL incentivize reasoning beyond the base model?").
- **Reasoning-Cache** (concurrent work) — one line.
