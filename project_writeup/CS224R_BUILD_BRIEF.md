# CS224R Deck — Build Brief / Handoff

This is a self-contained brief for building the project presentation in a fresh chat. It contains the storyline (the spec), the verified data, the figure recipes, the reference deck's style DNA, how McKinsey-style slides actually work, and the build/QA workflow. **Build to the storyline as a spec — every element listed for a slide must appear, and the four diagrams below must be built as diagrams, not retyped as text.**

---

## 0. FILES TO UPLOAD (start here)

**Must upload:**
1. **This file** (`CS224R_BUILD_BRIEF.md`) — the spec + data + style rules.
2. **`results_version_G.json`** — the run data; needed to regenerate every chart. (Section 4 + 5 tell you the exact keys.)
3. **`CS224R_Presentation.pptx`** — the *reference style deck* (its content is the old CS230 "Mockingbird" project; ignore the content, copy the **style**). The new builder must extract its real style DNA from the XML, not eyeball it. (Section 6 records the DNA, but uploading the file lets the builder verify.)
4. **`1780424537079_image.png`** — Chung's "Image 1": GSM8K clean-vs-mixed accuracy curves sitting on top of each other (slide 3 hook). *Can also be regenerated from the JSON — see Fig "Chung" in Section 5 — but uploading the original is safest.*

**Nice to have (deeper reading / verification, not strictly required):**
5. `RLAD_Training_LLMS...pdf` — the RLAD paper. The one table you need (Table 2) is already transcribed in Section 7, so you only need the PDF if you want to read around it.
6. `e3_Learning_to_Explore...pdf` — the e3 paper (the "low-KL → extrapolation" intuition this project complicates).
7. `Project_Milestone...vFinal.pdf` — the milestone (Chung's original framing/figure).
8. `1780434836724_image.png`, `1780435071776_image.png` — earlier MATH length plots (superseded by regenerated figures; reference only).

> Note: any figures/scripts from the previous chat live only in that sandbox and are **gone** in a new chat. Everything is regenerated from `results_version_G.json` using Section 5.

---

## 1. The one-sentence thesis

"Diversity" in RL-for-reasoning is **two knobs, not one.** Content-free **input noise** (random numeral-free trivia prepended to training prompts, **training-only**, eval always clean) **sharpens** the policy (entropy ↓); the **negative gradient** in e3 **explores** (entropy ↑). Distinct, even opposing. Trivia buys **efficiency at no in-distribution accuracy cost**; with e3 it sits at the **sweet spot** (most token-efficient *and* closest to base). Boundary condition: long-horizon out-of-distribution transfer (MATH → AIME), where the sharpening **breaks**.

**Title for the talk:** *Two Roads to Diversity: Input Noise Sharpens, Negative Gradients Explore.*
**Authors:** Anna Grebenchtchikova & Chung Wong. **Course:** Stanford CS224R, Spring 2026.

---

## 2. The 8-slide storyline (slides 0–7) — THIS IS THE SPEC

Build every listed element. Splits are written as (left ↔ right) fractions.

**Slide 0. Title slide.**

**Slide 1 — "Compute is the new gold — and 'diversity' is how RL spends it"** (1/3 ↔ 2/3)
Left third: the stakes — RL-for-reasoning explores, exploration is expensive (a simple **pictogram: $ / GPU → tokens → reasoning**). Right two-thirds: the one-line thesis stated up front — diversity is two knobs, not one: input noise sharpens, the negative gradient explores. Sets the whole deck. *(This is its own slide — do NOT fold it into the title slide.)*

**Slide 2 — "Today's recipe for 'diversity' gives inconsistent returns"** (1/2 ↔ 1/2)
Left: the RLAD inconsistency, shown as a **mini heat-strip or up/down arrows** (helps base ↑, hurts DAPO ↓, varies by benchmark) — make the messiness *visual*, not a numeric table. Right: the open question that falls out — how much is the content, vs. just perturbing the input? Motivation slide.

**Slide 3 — "A one-line trick matched accuracy at lower cost — so we stress-tested it"** (2/3 ↔ 1/3)
Left two-thirds: Chung's hook (Image 1 — the two accuracy curves on top of each other) with a "same accuracy" callout. Right third: the experiment grid as a **clean 3-block diagram** (GRPO clean/trivia · e3 clean/trivia/mixed · MATH clean/trivia), with the "step-limited, mixture not more data" note as a footnote strip.

**Slide 4 — "Trivia keeps accuracy and reaches peak on roughly half the tokens"** (3/4 ↔ 1/4)
Three-quarters: Fig 2 (the efficiency frontier) — hero figure. Right quarter: takeaway stack — accuracy preserved · fewer truncations · ~21M vs ~40M tokens-to-peak, as three stacked blocks with a down-arrow on "tokens."

**Slide 5 — "On MATH the model gets dramatically leaner for a 2-point price"** (3/4 ↔ 1/4)
Three-quarters: Fig 3 MATH panel (727 vs 1325). Right quarter: throughput-not-clock framing as a small block (same speed, fewer tokens → finished in budget; clean had to be stitched) + forward hook → content-aware (Fibonacci) trivia could do even better.
*AIME addition:* model becomes ~45% leaner in-distribution for ~2pp accuracy, **but AIME transfer collapses 6× (0.146 → 0.023)**. Wall-clock: MATH is the clean comparison — matched throughput; trivia finished in budget, clean exceeded the 24h cap and was stitched.

**Slide 6 — "The mechanism: trivia sharpens, e3 explores — opposite knobs"** (1/2 ↔ 1/2)
Left: Fig 4 (KL drift) — e3 stays closest to base, trivia+mixed closest of all. Right: the conceptual payoff as a **two-arrow diagram** (input noise → entropy ↓ / sharpen; negative gradient → entropy ↑ / explore), plus the honest twist as a one-line callout (low drift ≠ guaranteed generalisation — MATH-trivia stayed close yet transferred worst). Entropy lives here as **one sentence, not a chart**.
*AIME addition:* AIME (GSM8K-trained) bars show differences **within error bars** — the only clear lever is the e3 gradient. The twist/counterexample: MATH-trivia drifts least from base yet transfers worst (AIME 0.023) → **low KL is necessary, not sufficient** for generalization.

**Slide 7 — "Together they're the sweet spot — and here's where it goes next"** (2/3 ↔ 1/3)
Left two-thirds: synthesis — e3 + trivia = most token-efficient AND closest to base, shown as a **small 2×2 or a converging-arrows visual**. Right third: future work as an **arrow timeline** (domain trivia on MATH → more AIME conditions → scale to o4-mini, tying to RLAD's weak-to-strong).

---

## 3. The four elements that MUST be built as DIAGRAMS (not text)

The previous attempt failed mainly by retyping these as text blocks. Build them as graphics:
1. **Slide 1 pictogram:** `$ / GPU → tokens → reasoning` — three icons/labels joined by arrows. (Make the "compute spent on exploration" idea legible at a glance.)
2. **Slide 2 heat-strip / arrow matrix:** a small grid (rows = base / DAPO / RLAD; cols = AIME / DeepScaleR / AMC) coloured or arrowed by whether abstractions help (↑/green) or hurt (↓/red). The point is the *messiness*, visually. (Numbers live in Section 7 if you want them on hover/caption — but the visual is the deliverable.)
3. **Slide 3 three-block diagram:** three labelled blocks — GRPO {clean·trivia}, e3 {clean·trivia·mixed}, MATH {clean·trivia} — as a clean experiment grid, with the step-limited footnote.
4. **Slide 7 converging-arrows (or 2×2):** two inputs (efficiency from trivia, accuracy from e3) converging into "sweet spot," or a 2×2 placing e3+trivia in the winning quadrant.

---

## 4. Verified data (source of truth)

7 training runs, all **400 steps**, batch 64, rollout.n 8 → **25,600 samples each**. **Step-limited**: the "2M mixed" run is a **50/50 batch mixture** of clean+trivia prompts — *not* more compute or more data. Never imply otherwise.

**Track ↔ run-name map** (run names are the JSON keys, see §5):
| Track | Meaning | JSON run key |
|---|---|---|
| A | GRPO clean | `gsm8k_clean` |
| B | GRPO trivia | `gsm8k_trivia_only` |
| C | e3 clean (partial e3) | `gsm8k_partial_e3_clean` |
| D | e3 trivia (partial e3) | `gsm8k_partial_e3_trivia` |
| E | e3 mixed (2M, partial e3) | `gsm8k_track_e_2M_partial_e3` |
| — | MATH clean | `math_clean` (stitched) |
| — | MATH trivia | `math_trivia_only` |

**GSM8K val accuracy — final | peak:** A .861|.862 · B .846|.871 · C .883|.889 · D .883|.883 · E .878|.881. (Trivia leads clean at steps 100–300, peaks higher; e3 adds ~2pp.)

**MATH val accuracy — final | peak:** clean .724|.733 · trivia .686|.711 (unstable).

**AIME 2025 pass@1** (n=8 samples, 60 problems, SE ≈ 0.05): A .173 · B .154 · C .217 · D .196 · E .202 · MATH-clean .146 · MATH-trivia .023. GSM8K AIME differences are **within noise**; the e3 lift (A→C +.044) is the only clear effect. MATH-trivia = real **6× collapse** (lowest extract-fail, 18%).

**Tokens-to-90%-of-peak (millions, hardware-independent):** D 20.8 (best) · C 29.6 · E 30.4 · A 39.3 · B 42.1.

**Val response length** (overall key, excl `0_/1_length`): GSM8K final A 587 · B 537 · C 630 · D 668 · E 665 → **small, curves cross** (NOT a clean effect). MATH **1325 (clean) vs 727 (trivia) ≈ 45% leaner** — the real effect. *(An earlier 811/496 figure was a bug from keys ending `length/mean`.)*

**KL drift to base — final | mean:** A .510|.190 · B .454|.228 · C .218|.147 · D .163|.110 · E .182|.111 · MATH-clean .288|.117 · MATH-trivia .105|.107. Robust: **all e3 ≪ both GRPO**; within e3, D & E closest. MATH-trivia lowest KL yet worst OOD — the counterexample.

**Truncation:** GSM8K 16% → 14%; MATH 36% → 29% (trivia lower). **Trivia prompt overhead:** +7.3 tokens. **Entropy:** GRPO/trivia collapse (.019/.025); e3 high (.13–.18) — trivia sharpens (entropy ↓), e3 negative gradient explores (entropy ↑).

**Wall-clock:** trivia consistently faster; MATH is the clean matched-throughput case (~6.1–6.5k tok/s); clean MATH exceeded the 24h cap and was **stitched**. W&B doesn't log runtime cleanly → **lead efficiency with tokens-to-peak; use wall-clock only via the MATH "finished in budget vs stitched" line.**

---

## 5. Figures — what each shows + exact recipe to regenerate from the JSON

Data lives at `data['training'][run_key]['rows']`; each row has `_step` plus metric keys. AIME lives at `data['aime'][label]['pass@1']`.

**Metric keys inside each row:**
- entropy → `actor/entropy`
- KL to base → `actor/kl_loss`
- train response length → `response_length/mean`
- **val accuracy** → any key starting `val/` and containing `test_score`
- **val length** → key starting `val/`, ending `/length/mean`, **excluding** `0_length` / `1_length`

**AIME labels** in `data['aime']`: `Track A GSM8K`, `Track B GSM8K`, `Track C GSM8K (partial e3 clean)`, `Track D GSM8K (partial e3 trivia)`, `Track E GSM8K (2M mixed partial e3)`, `Track A MATH`, `Track B MATH`. Use SE = sqrt(p(1−p)/60) for error bars.

**Figures to produce (restyle into deck palette — Liberation Sans / Arial-metric, white bg):**
- **Fig 2 — efficiency frontier (hero, slide 4):** GSM8K val accuracy (y) vs **cumulative tokens generated** (x = cumsum(64·8·response_length)/1e6), all 5 GSM8K tracks; ring each run at 90%-of-its-peak. Annotate "e3+trivia: peak on ~half the tokens."
- **Fig 3 — response length (slide 5):** MATH panel, clean vs trivia (1325 → 727, ~45% shorter). (A two-panel GSM8K+MATH version exists; slide 5 uses MATH only.)
- **Fig 4 — KL drift (slide 6):** GSM8K + MATH panels; smooth with a centered rolling mean (window ~7, min_periods=1) to keep edges. All e3 below both GRPO.
- **AIME bars — GSM8K (slide 6):** pass@1 for tracks A–E with SE error bars; caption "within error bars."
- **AIME bars — MATH (slide 5 AIME add):** clean .146 vs trivia .023 with SE; annotate "6× collapse."
- **Chung (slide 3):** clean (`gsm8k_clean`) vs mixed (`gsm8k_track_e_2M_partial_e3`) val accuracy curves overlapping; "same accuracy, lower cost."
- **Entropy (optional, slide 6 one-liner only):** entropy-vs-accuracy phase; GRPO/trivia collapse, e3 stays high. *Use as a sentence, not a hero chart.*

**Colour encoding (cool = clean, warm = trivia) — keep consistent everywhere:**
A GRPO clean = navy `#051C2C` · B GRPO trivia = orange `#FFA800` · C e3 clean = blue `#134F78` · D e3 trivia = red `#CD3030` · E e3 mixed = mid-blue `#0679C3` (dashed).

---

## 6. Reference deck style DNA (extracted from `CS224R_Presentation.pptx`)

- **Canvas:** 13.333 × 7.5 in (16:9).
- **Fonts:** **Georgia** (titles, bold) · **Arial** (body).
- **Palette:** deep navy **`#051C2C`** (dominant panel motif), accent blue **`#134F78`**, light-blue fill **`#D5EBFA`**, mid-blue `#0679C3`, sharp orange **`#FFA800`**, red **`#CD3030`** (✗), green ✓, white.
- **Action titles:** top-left, Georgia bold, ~24–44pt, black on white / white on navy, may wrap to 2 lines (never 3 — shorten or widen).
- **Recurring motif:** a **navy side panel** (¼, ⅓, or ½ width) carrying the takeaway / "so what" in white text. Icons in circles; `→` arrows; blue header tables; page number bottom-right; small italic source line.
- The reference deck does asymmetric splits (¼-¾, ⅓-⅔, ½-½) exactly as the storyline specifies — reuse them.

**Always**: re-extract the real DNA from the uploaded `.pptx` XML (fonts, sizes, colours, panel widths) before generating — don't approximate.

---

## 7. RLAD Table 2 (Qwen3-1.7B) — for slide 2's heat-strip

Columns = three benchmarks × {w/o abs, w/abs avg, w/abs best}. Arrows compare **w/o → w/abs (avg)**: ▲ helps, ▼ hurts.

| Approach | AIME w/o | AIME w/abs | AIME best | DSR w/o | DSR w/abs | DSR best | AMC w/o | AMC w/abs | AMC best |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 33.75 | 36.25 ▲ | 40.00 | 20.21 | 22.14 ▲ | 32.50 | 86.41 | 78.01 ▼ | 84.53 |
| + DAPO | 37.92 | 34.90 ▼ | 39.79 | 21.67 | 21.88 ▲ | 33.54 | 86.41 | 81.99 ▼ | 88.44 |
| + RLAD | 38.04 | 42.45 ▲ | 48.33 | 23.54 | 24.84 ▲ | 35.54 | 87.25 | 88.35 ▲ | 91.72 |

Story for the heat-strip: abstractions help the base on AIME/DSR but **hurt on AMC**; **hurt DAPO on AIME**; help RLAD everywhere — and gains *without* abstractions vary widely. Source: Liu et al., *RLAD* (2025), Table 2.

---

## 8. How McKinsey-style slides work (so the builder gets the form right)

- **Action titles, not topic labels.** The title is the *takeaway as a full sentence* ("Trivia keeps accuracy and reaches peak on ~half the tokens"), not a category ("Results"). A reader should grasp the argument from the titles alone.
- **Horizontal logic.** Read top-to-bottom, the titles form a coherent argument: stakes → inconsistency → hook → efficiency → leanness → mechanism → synthesis. Check this reads as a sentence.
- **One message per slide.** Everything on the slide supports that single message; if a chart doesn't serve the title, cut it.
- **Asymmetric splits + the "so-what" panel.** Evidence (chart/diagram) takes the big side; the narrow navy panel carries the implication. Splits vary by slide (⅓-⅔, ½-½, ¾-¼, ⅔-⅓) — as specified per slide.
- **Every slide is visual.** Charts, pictograms, diagrams, stat callouts — not bullet walls. Big-number callouts (e.g., **~21M**, **6×**, **≈45%**) earn their place.
- **Vertical logic within a slide.** Lead with the point; supporting detail beneath. Left-align body; centre only titles.
- **Quiet chrome.** Consistent footer, page numbers, small italic source citations. No decorative full-width bars, no accent lines under titles.
- **The golden thread.** Keep asking "what would the audience miss if this slide/element were removed?" If nothing — cut it.

---

## 9. Build & QA workflow

- **Build:** create from scratch in **pptxgenjs** (`LAYOUT_WIDE`/13.333×7.5), or use the reference `.pptx` as a template and replace content — either is fine, but match §6 DNA exactly. Charts as restyled matplotlib PNGs placed via `addImage` (exact palette control); the RLAD heat-strip and the four diagrams built as native shapes/SVG.
- **Render to images for QA:** convert the `.pptx` → PDF (LibreOffice/`soffice`) → JPGs (`pdftoppm`), then **visually inspect every slide**.
- **Look for:** 3-line titles colliding with content (keep titles ≤2 lines), text overflowing boxes, chart captions overlapping chart axis labels, elements < 0.5" from edges, low-contrast text on navy.
- **Then re-check against the storyline:** for each slide, confirm *every* listed element is present (this is where the last attempt failed — it dropped slide 1 entirely and substituted text for the four diagrams).

---

## 10. Framing rules / things to get right (non-negotiable)

- **Step-limited** everywhere: "mixed/2M" = 50/50 batch mixture, **not** more compute/data.
- **AIME is secondary / "nice to have"**, always **with error bars** (SE ≈ 0.05, 60 problems). Don't over-claim GSM8K AIME differences — within noise.
- **Efficiency = tokens-to-peak** (hardware-independent). Wall-clock only via the MATH "finished in budget vs stitched" line.
- **Compute-as-gold** = motivation/north-star, **not** a proven result.
- **The twist** (low KL ≠ guaranteed transfer; MATH-trivia closest to base yet worst on AIME) is the intellectual payoff of slide 6 — keep it.
- **Entropy** = one sentence, not a figure.
- **No fabricated links/URLs.** If a link can't be verified, leave it blank and say so.

---

## 11. Papers (cite, with weight)

- **e3** (Setlur et al., 2025) — deep read; the gradient-side / exploration comparison and the "low-KL → extrapolation" intuition this project complicates.
- **RLAD** (Liu et al., 2025) — deep read; Table 2 (Section 7) is slide 2's motivation; content-rich abstractions are the contrast to content-free trivia.
- **DAPO** (Yu et al., 2025) — one line (baseline in the RLAD table).
- **Yue et al. (2025)** — one real sentence: sharpening vs exploration ("does RL incentivize reasoning beyond the base model?").
- **Reasoning-Cache** (concurrent work) — one line.
