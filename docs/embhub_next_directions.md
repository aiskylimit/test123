# EmbHub — Next Directions

Concise spec of what to try next, and why. Read "Why we're here" and "Guiding insight"
first — they determine the order.

## Why we're here (the controlled negative)

From-scratch, EMBEDDING-layer, additive EmbHub is a CLEAN NEGATIVE, baseline-controlled:

- Trained baseline (no hub) + S3 hub at alpha 0.15 and 0.20, identical config, to step 6500.
- **Test B (embedding cosine, translation vs random) at alpha=0: baseline +0.0504 >= hub
  +0.046.** The base model aligns languages ON ITS OWN; the hub is slightly BELOW baseline.
- Adding hub contribution (alpha 0 -> 0.3) DECREASES the gap — the contribution slightly
  DILUTES alignment rather than adding to it.
- Bigger training alpha (0.20) did not fix it: the hub is now load-bearing (norm_ratio ~9%,
  0% dead anchors, no LM-loss cost) yet still does not bridge languages.
- Test A (anchor-weight overlap) shows a small significant gap (~+0.013 JS, p=1e-3) but it is
  functionally inconsequential — Test B shows it does not produce more aligned embeddings.
- Training longer does not help (tested).

## Guiding insight (why the data is fine, and what the hub must do)

The data is NOT the problem. The baseline Test B IMPROVES over training, which proves the
corpus already contains enough NATURAL cross-lingual signal. A concrete source of this: many
FREQUENT words in the small languages are literally English words appearing inside
small-language text (brand names, technical terms, loanwords, code, named entities) — the
MUSE dictionary check showed a large fraction of small-language entries are IDENTICAL to their
English form. So the training data already contains a form of NATURAL, incidental
code-switching: English tokens occurring in Vietnamese/Arabic/etc. contexts. Together with
cognates, shared numerals, and shared named entities, this is enough signal that a from-scratch
model extracts cross-lingual alignment into its base embeddings on its own.

Therefore the goal of the architecture-only route is NOT "create cross-lingual structure from
nothing" — it is "extract MORE of the naturally-present signal into the anchors than the base
model already puts into its embeddings, WITHOUT translation data."

The hard constraint this creates: the natural signal flows into the base embeddings whether or
not the hub exists. So for the hub to BEAT baseline, the anchors must capture cross-lingual
structure the base embeddings DO NOT already hold. An embedding-layer anchor block operating
in token space competes directly with embeddings that already have this signal -> it keeps
being REDUNDANT with them (exactly the observed ~baseline result). The way to capture
something NON-redundant is to put the anchors where different structure lives (deeper layers)
and/or let them encode a different space (decoupled keys/values, transform).

## Order to try (decided)

1. **Architecture-only, from-scratch, NO translation data** — test whether a better
   architecture can pull out the natural signal. Lead with MID-LAYER placement + decoupled
   keys/values (the changes that can capture NON-redundant structure), not just a fancier
   embedding-layer combination.
2. **Objective + architecture** (and optionally objective + OLD architecture as a control, to
   isolate the objective's own contribution) — salvage route if (1) stays ~baseline. Adds
   translation data, so it spends the "no parallel data" advantage.
3. **Finetuning a pretrained model** — highest-probability positive; held last by choice.

Mechanism refinements (top-k, fewer anchors, fancier similarity) are LAST and only after a
positive result — they optimize a working mechanism, they do not create the signal.

---

## 1. ARCHITECTURE-ONLY, FROM-SCRATCH  [try first]

Goal: extract the naturally-present cross-lingual signal into the anchors WITHOUT translation
data, and beat the no-hub baseline. The primary variants are V2, V3, V4, V5 (V2 also has cheap
sub-variants V2b and V2c/+tail/+buckets). Run each as a separate experiment, each with its own
matched no-hub baseline. See "Suggested run order" at the end of this section for which to run
first — you do NOT need to run all of them.

Shared elements across ALL variants (V2-V5):
- SELECTION is always the validated cosine + learnable-temperature rule:
  `w = softmax(cos(x, keys) * scale)`, temperature as a learnable log-scale (init log(14),
  LR 75x, excluded from weight decay). Keep this identical in every variant — it is the one
  piece already known to work; the variants change only what happens AFTER selection (how the
  retrieved mixture is combined with `x`) and WHERE the block sits.
- `x` = the representation the block operates on: the input token embedding for V2-V4, a
  mid-layer hidden state for V5. In all cases `keys`/`values` are the anchors (decoupled into
  separate key/value sets from V3 onward).
- SAFE INIT is mandatory in every variant (each variant states its exact form below). The
  block must start as `output ~= x` with ~zero anchor contribution, so step 0 does not corrupt
  training. Verify per the VERIFICATION checklist below (one-off step-0 pass-through check, then
  watch loss-vs-baseline through ~1000 steps — 100 steps only catches gross breakage).
- MEASUREMENT/verdict is the same in every variant: the anchor-layer Test B vs the matched
  no-hub baseline (does the hub make translation pairs' representations AT THE ANCHOR LAYER
  more similar than the no-hub baseline does?). Test A (anchor overlap) is secondary. The
  existing embedding-additive result (baseline +0.0504 >= hub +0.046) is the number to beat.

LEARNING RATE — CRITICAL (do not get this wrong):
- The 75x LR multiplier applies ONLY to the temperature parameter `log_logit_scale`. It exists
  because that single scalar has a tiny gradient and would otherwise stay frozen.
- Do NOT put the contribution parameters on 75x. `W` / `W_mix` (V2), `Linear_v`, `Linear_g`,
  `anchor_keys`, `anchor_values` (V3-V5) all stay on the NORMAL base LR (same as the rest of the
  model). Boosting them makes the anchor contribution grow FASTER than the model can adapt and
  re-creates, a few hundred steps in, exactly the disruption safe init was meant to prevent
  (just delayed instead of at step 0).
- Recommended optimizer param groups: (group 1) `log_logit_scale` — base LR x 75, weight decay
  0; (group 2) everything else including all anchor/linear/gate params — base LR, normal weight
  decay. That is the whole special-casing; nothing else gets a boosted LR.
- After safe init, the contribution grows GRADUALLY on its own: its gradient is proportional to
  how much the anchors are helping the loss, so it grows only as fast as the anchors become
  useful (in V3 the near-closed gate throttles it further — a self-limiting ramp). This is the
  intended behavior; do not try to speed it up with a higher LR.

VERIFICATION — a short CHECKLIST, not a system to build (these use quick checks / existing
metrics, no new infrastructure):
1. (implementation requirement) STEP-0 CHECK: after writing the safe init, run a one-off check
   that the hub block is a pass-through — feed a random `x`, assert `block(x) == x` to tolerance.
   ~5 lines, runs in milliseconds on CPU, no training run needed. Do it in fp32 (or use a loose
   tolerance ~1e-2 in bf16, since bf16 has only ~3 significant digits and a tight 1e-5 would
   false-alarm). This is a checklist item to confirm the init once, not code to add to training.
2. (just monitor existing metrics) EARLY LOSS: when the run starts, watch that the hub variant's
   loss TRACKS the matched no-hub baseline (same data + seed) over the first steps. Loss is
   already logged — nothing to implement. If it does not track, safe init is wrong (the step-0
   check should have caught it) or the contribution is growing too fast (put contribution params
   on base LR not 75x; for V3 make the gate bias more negative). Note ~100 steps only catches
   GROSS breakage; a SLOW divergence may not show until ~500-1000 steps, so keep watching loss
   vs baseline through ~1000 steps before trusting the run.
3. (V2c family only, just monitor) DEAD ANCHORS: watch the existing `dead_anchor_frac` metric —
   it must not climb alarmingly. Hard top-k routing is active from step 0 (the linear safe init
   does NOT neutralize it — see the V2c safe-init note), and anchor death is a SLOW process
   (thousands of steps), so this is a trend to watch over ~1000+ steps, not a step-0 event.
Summary: check (1) once at implementation time; watch (2) and (3) — both already-logged metrics
— over the first ~1000 steps before committing to a full run. Nothing here needs new code beyond
the one-off step-0 check.

### Variant V2 — CONCAT + linear  [your original concat idea]
Replace the additive mix with concat-then-linear. The linear layer LEARNS the mixing ratio, so
there is NO alpha (a fixed alpha in front of a learnable linear is redundant — the linear can
absorb any constant scale into its own weights).
```
mixture = softmax(cos(x, anchors) * scale) @ anchors      # keys = values = anchors
output  = Linear([x ; mixture])                           # concat (2d) then one linear (2d -> d)
```
Why concat+linear can help where additive did not: plain-add forces the anchor contribution to
live in the SAME space/direction as x and only push it around additively (which is why it
DILUTED alignment). The linear can TRANSFORM the concatenated vector — project, rotate, or
down-weight the anchor part per dimension — so the model can learn to use the anchor signal in
a useful direction instead of being forced to add it raw.

SAFE INIT for V2 (do not skip — this replaces the "small alpha keeps step 0 safe" property):
The linear maps a 2d vector `[x ; mixture]` to d. Write its weight as two horizontal blocks,
`W = [W_x | W_mix]` where `W_x` is d x d (acts on x) and `W_mix` is d x d (acts on mixture).
Initialize `W_x = Identity` and `W_mix = 0` (and bias = 0). Then at step 0,
`output = I*x + 0*mixture = x` exactly — the block starts as a pass-through of x, contributes
nothing from the anchors, and training grows `W_mix` away from zero only as the anchors become
useful. Without this, a randomly-initialized linear scrambles x at step 0 (an untrained dense
map of the embedding) and corrupts early training. (Verify per the VERIFICATION checklist:
step-0 pass-through check, then loss-vs-baseline through ~1000 steps.)

#### V2 sub-variants (small changes to the combine step — cheap to try alongside V2)

**V2b — transform the mixture with an MLP before concat.**
`output = Linear_out([x ; GELU(Linear_v(mixture))])`. NOTE: putting a PLAIN linear on the
mixture before the concat adds nothing — two stacked linears with nothing between them collapse
into one, and the outer `Linear_out` can already represent any linear transform of the mixture.
It only becomes more expressive with a NONLINEARITY between them (the GELU above), i.e. the
mixture passes through a small 2-layer MLP before combining. Worth trying if plain V2 shows
partial signal. Safe init: init `Linear_out` as `[I | 0]` (as in V2); the pre-MLP can be
ordinary init since the `[I | 0]` outer linear already forces `output ~= x` at step 0.

**V2c — concat the TOP-K anchor vectors (not the weighted sum).**
Instead of averaging the selected anchors into one vector, keep the k most-similar anchors
SEPARATE and hand them all to the linear:
`output = Linear([x ; w1*a_{i1} ; w2*a_{i2} ; ... ; wk*a_{ik}])`, where i1..ik are the top-k by
cosine similarity (k = 5 or 10) and w1..wk are their selection weights (see "weighting" below).
Input dim is (k+1)*d.
Why it differs from V2: the weighted sum AVERAGES the top anchors into one d-dim vector and
loses which-anchor-contributed-what; concatenating keeps them distinct, so the linear sees each
retrieved anchor individually and can combine them per-slot. More information than the weighted
sum, and closer to how retrieval/memory architectures use a SET of retrieved items.

Weighting each concatenated anchor (which scalar to multiply by) — three options:
- Do NOT use raw cosine similarity as the multiplier: in this model cosines are small and
  clustered (random ~+-0.03; even trained top anchors only ~0.1-0.3), so multiplying by raw
  cosine shrinks the anchors ~5-7x and the raw range does not reflect relative selection
  strength well.
- OPTION B (try FIRST): use each top anchor's RAW softmax weight (the softmax over all N,
  so the top-k weights sum to <1 — larger when selection is sharp, smaller when diffuse). This
  PRESERVES absolute selection confidence: a decisive selection gives large weights, an
  uncertain one gives small weights (and small early-training weights are fine — safe init
  already keeps the contribution ~0 until selection sharpens). Most information, no extra step.
- OPTION A (fallback): RENORMALIZED top-k weights (softmax weights of just the top-k, divided
  by their sum so they sum to 1). Gives clean relative ranking within the top-k, BUT discards
  absolute confidence (a sharp vs diffuse selection can look identical after renormalizing) —
  which is exactly the signal weighting was meant to add. Use only if Option B's contribution
  turns out too weak in practice (selection stays diffuse late in training).
- OPTION C (no weighting): plain unweighted concat, `[x ; a_{i1} ; ... ; a_{ik}]` (all
  multipliers = 1). Simplest, but carries the LEAST information — the linear sees each top
  anchor vector but not how strongly it was selected, and cannot recover per-token confidence
  on its own (it can only learn fixed per-slot weights). Reasonable as a baseline to check
  whether weighting matters at all.
Prefer Option B first; A is the rescue if B's scale is too weak; C is the no-weighting baseline.

Caveats: (i) top-k is a HARD selection, so gradient flows only to the k chosen anchors per token
(like MoE hard routing) — reintroduces some dead-anchor risk that full softmax avoids (the
V2c+tail variant below fixes this); (ii) the concat imposes an order (by similarity rank), so
slot j = "the rank-j anchor" — fine, but sensitive to ties and makes the input (k+1)*d wide.
Safe init: init the linear so the x block is Identity and ALL anchor-slot blocks are 0
(`W = [I | 0 | 0 | ... ]`) -> `output ~= x` at step 0.

**V2c+tail — top-k concat PLUS one aggregated "rest" slot (anti-collapse).**
Pure V2c gives gradient only to the k selected anchors, so the other ~N-k can die. Fix: also
weighted-sum ALL the non-top-k anchors into one extra d-dim slot and concat it:
`output = Linear([x ; w1*a_{i1} ; ... ; wk*a_{ik} ; mixture_rest])`, where
`mixture_rest = softmax-weighted sum of the non-top-k anchors`. Every non-top anchor now gets
SOME gradient through that aggregated term, keeping it trainable (the full-softmax cushion that
pure top-k loses). The point of the tail slot is not its content (non-top anchors are weakly
matched) but keeping all anchors ALIVE. Cheap: one extra slot, input becomes (k+2)*d.
Weighting: use the SAME choice as V2c for the top-k slots (Option B first). The tail slot itself
is a full softmax-weighted sum over the non-top anchors (its weights are the softmax weights of
those anchors, i.e. no separate choice needed). Safe init: same `[I | 0 | ... | 0]` (identity on
x, zero on every anchor slot INCLUDING the tail slot).

**V2c+buckets — graded tail (deferred refinement of V2c+tail).**
Instead of ONE "rest" slot, split the non-top-k anchors into ~10 buckets BY SIMILARITY RANK
(e.g. ranks 11-100, 101-200, ...), weighted-sum each bucket, and concat those bucket vectors.
Gives the linear a coarse graded summary of which similarity-band was active, and still spreads
gradient to all anchors. IMPORTANT: bucket by similarity RANK, not by anchor index — index
buckets average unrelated anchors and carry no signal. Adds width/complexity for marginal gain
over V2c+tail; DEFER — only try if V2c+tail helps and you want a graded tail.
Weighting: same as V2c for the top-k slots (Option B first); each bucket slot is the softmax-
weighted sum of the anchors in that rank band. Safe init: same identity-on-x, zero-on-ALL-slots
(top-k slots AND every bucket slot).

SAFE-INIT NOTE for the whole V2c family (V2c / +tail / +buckets): the `[I | 0 | ... | 0]`
linear init guarantees `output ~= x` at step 0 (identity on x, zero on every anchor/tail/bucket
slot), so early training is not corrupted — same guarantee as V2. BUT unlike V2/V3, this does
NOT neutralize the anchor ROUTING: top-k selection is a HARD choice active from step 0, decided
by the random initial cosine similarities, so which anchors get gradient is already being
gated before the linear has learned anything (the dead-anchor dynamics start immediately). The
zero anchor-blocks make the CONTRIBUTION ~0 but do not make the routing uniform. This is the
reason V2c+tail exists (the tail slot keeps non-top anchors trainable). If early routing
instability is a problem, options: warm up with full softmax (no top-k) for the first N steps
then switch to top-k, or start with a larger k and anneal down. (Verify per the VERIFICATION
checklist: watch loss-vs-baseline AND dead_anchor_frac over ~1000 steps — anchor death is slow,
so 100 steps will not reveal it.)

### Variant V3 — upgraded anchor block (upgrades 1+2+3, combined)  [upgrades on V2]

THE IDEA IN ONE SENTENCE: instead of the anchors being one set of vectors that are matched,
retrieved, and added raw (V1/V2), give the retrieval three degrees of freedom — separate
"address" vs "content" per anchor, a learned transform on the retrieved content, and a learned
per-dimension on/off switch for how much to add — so the hub can store, shape, and inject
cross-lingual information that the base embeddings do NOT already contain.

Walk through what each upgrade fixes, in the order the data flows:
- When a token selects anchors, WHAT it matches against and WHAT it gets back are currently the
  same vector. Upgrade (1) splits them: a `key` for matching, a separate `value` for the
  content returned. Now an anchor can be "the anchor that Chinese tokens point at" while the
  content it returns is a shared cross-lingual vector — and the value can live in a different
  space than the token, so it is not forced to just re-encode the embedding.
- The retrieved content (`mixture`) then gets a learned linear transform, upgrade (2), which
  rotates/projects it into whatever direction actually helps before it touches the token.
  (In V1/V2 the retrieved vector was added essentially raw, pointing wherever it happened to
  point — which is why it diluted alignment.)
- Finally, upgrade (3) replaces the single global `alpha` with a learned per-token,
  per-dimension gate that decides how much of the transformed content to admit — open where the
  anchor helps, closed elsewhere.

IMPORTANT — run upgrades 1+2+3 TOGETHER as one architecture; do NOT test them separately first.
They remove three complementary bottlenecks (HOLD non-redundant content / POINT it the right
way / APPLY it selectively); any one alone leaves the others blocking, so it would underperform.
They are individually toggleable only so that IF V3 beats baseline you can ablate which one
mattered afterward — a post-success analysis, not the first run.

Form (gate + add is preferred — more targeted than concat, and naturally recovers "removable at
inference" because the gate learns to close where anchors do not help):
```
keys, values = anchor_keys (N x d), anchor_values (N x d)  # (1) decoupled key/value
w       = softmax(cos(x, keys) * scale)
mixture = w @ values
update  = Linear_v(mixture)                                # (2) transform anchor content (d x d)
gate    = sigmoid(Linear_g(x))                             # (3) per-token, per-dim gate (replaces alpha)
output  = x + gate * update
```

The three upgrades in V3 (run together):

- **(1) Decoupled keys/values.** Give each anchor a separate `key` (used for selection) and
  `value` (used for the contribution), instead of one vector serving both roles. This lets an
  anchor be ADDRESSED by (say) Chinese tokens while CONTRIBUTING a shared-meaning vector, and
  lets the values encode a DIFFERENT space than x — so the hub is not forced to re-encode what
  the base embeddings already hold (directly targets the "redundant with embeddings ->
  ~baseline" failure). Cost: doubles anchor params (N x d keys + N x d values).

- **(2) Transform the anchor content (`Linear_v`, d x d).** Map the retrieved mixture into a
  useful direction BEFORE it touches x. Plain-add could only add the raw mixture (which pointed
  the wrong way and diluted alignment); a learned projection lets the model send the anchor
  content wherever it actually helps. Cheap, high value.

- **(3) Per-dimension gate (`Linear_g`, d x d -> sigmoid) — REPLACES alpha.**
  `gate = sigmoid(Linear_g(x))`, `output = x + gate * update`. Instead of one global scalar
  alpha, the model learns a per-TOKEN, per-DIMENSION value in [0,1] controlling how much anchor
  signal to admit. Directly targets the "uniform dilution" failure — the model opens the gate
  for tokens/dims where the anchor helps and closes it elsewhere. Strictly more expressive than
  alpha; do NOT also keep a fixed alpha (redundant).

Alternative combination form (instead of gate+add): `output = x + Linear([x ; update])`
(concat the transformed update with x, then linear). Equivalent expressiveness for the
combination step; gate+add is preferred for the reasons above.

SAFE INIT for V3 (do not skip): start as `output ~= x` with ~zero anchor contribution, then let
training grow it. Set BOTH of the following (not just one):
- initialize `Linear_v` weights to ~0 (or equivalently `anchor_values` to ~0) so `update ~= 0`
  at step 0, AND
- initialize `Linear_g`'s bias to a strongly NEGATIVE value (e.g. -5) so
  `gate = sigmoid(-5) ~= 0.007 ~= 0` at step 0.
Why both, not either: mathematically either alone already gives `output ~= x`. But if you zero
only `Linear_v` and leave the gate wide open, the moment `Linear_v` moves the full-strength gate
lets it through abruptly (no gentle ramp); and if you set only the gate bias negative but leave
`Linear_v` random, the near-zero gate is multiplying a LARGE random update, so tiny gate
fluctuations inject noise. Setting both makes the contribution doubly ~0 at step 0 and grow
smoothly. (If you use the concat-form alternative `x + Linear([x ; update])`, init that linear
as `[I | 0]` like V2 instead.) Without safe init the untrained block scrambles x at step 0 and
corrupts early training. (Verify per the VERIFICATION checklist: one-off step-0 assertion that
`output == x`, then loss-vs-baseline through ~1000 steps.)

### Variant V4 — V3 + multi-head retrieval  [upgrade 4, on top of V3]

THE IDEA: V3 does ONE similarity comparison over the whole d-dimensional vector, so a token
retrieves anchors based on its overall direction. Multi-head splits the vector into h pieces
(heads) and does a SEPARATE retrieval within each piece, then concatenates the results. This is
the same trick as multi-head attention: different heads can specialize on different aspects —
e.g. one head routes on meaning, another on syntax or script — so the selection is finer-grained
than a single whole-vector match. Everything after selection (transform, gate, add) is unchanged
from V3. Only run V4 if V3 already shows signal; it adds parameters and complexity, so it is a
step UP from V3, not a replacement.
```
# split d into h heads; each head has its own keys/values and does its own cosine selection:
for each head i:  w_i = softmax(cos(x_i, keys_i) * scale_i);  mix_i = w_i @ values_i
mixture = concat(mix_1 .. mix_h)                          # then identical to V3:
update  = Linear_v(mixture)
gate    = sigmoid(Linear_g(x))
output  = x + gate * update
```
Here `x_i` is the i-th slice of x (size d/h), and `keys_i`/`values_i` are that head's anchor
key/value matrices of shape (N, d/h) — i.e. each head has its own N anchors living in its own
d/h-dim subspace, so total key params are h * N * (d/h) = N * d, the same budget as V3. Concat
of the h per-head mixtures (each d/h) gives a d-dim `mixture`, so `Linear_v` stays d -> d exactly
as in V3. Cost: the per-head bookkeeping and h separate cosine/softmax ops. SAFE INIT: same as
V3 (`Linear_v` ~0 or `anchor_values` ~0, AND `Linear_g` bias strongly negative -> `output ~= x`
at step 0).

### Variant V5 — MID-LAYER placement  [the key lever; place the V3 (or V4) block on a mid layer]
Put the anchor block (the V3 block, or V4 if multi-head helped) after a MIDDLE transformer
layer (~layer 6-14 of 28) instead of at the embedding. `x` is then the mid-layer hidden state.
Rationale: at the embedding layer the anchors see only token identity (easiest signal =
language identity) and compete with embeddings that ALREADY hold the embedding-level
cross-lingual signal -> redundant -> ~baseline (the observed result). Mid-network is where
multilingual models form language-AGNOSTIC representations, so anchors there attend over
already-partially-aligned states AND can capture DEEPER alignment the base EMBEDDINGS do not
hold — i.e. NON-redundant structure the hub can actually add. This is the variant most likely
to beat baseline architecture-only. Contribution is added to the hidden state at that layer;
rest of the model unchanged. Optionally place blocks at multiple depths.
Safe init for V5: same as the block it uses (V3/V4 init: `Linear_v` ~0 or `anchor_values` ~0,
`Linear_g` bias strongly negative) so it starts `output ~= x`.

### MEASUREMENT (applies to EVERY variant)
If anchors live at layer L, their effect is in the LAYER-L hidden states, not the input
embeddings. So run the Test-B-style comparison on layer-L representations: do translation
pairs have more similar layer-L states WITH the hub than the no-hub baseline at layer L? For
V2-V4 (embedding layer) L = the embedding output; for V5, L = the mid-layer the block sits on.
Measuring only input-embedding Test B for a MID-layer block (V5) would MISS its effect. Keep
Test A (anchor overlap) as a secondary signal; the verdict is anchor-layer Test B vs the
matched no-hub baseline.

Suggested run order for step 1 (do NOT run everything — this is a two-run first pass):
1. RUN V5 FIRST (the V3 block placed on a mid layer, ~layer 10). It is the highest-probability
   positive WITHIN architecture-only (finetuning, Section 3, is higher-probability overall but is
   deliberately held last), because V5 is the only variant that puts the anchors where
   NON-redundant structure lives. If V5 beats baseline you may not even need the embedding runs.
2. THEN RUN V3 (at the embedding layer) as confirmation. If V5 lost, V3 tells you whether the
   embedding layer is also dead (expected) before declaring architecture-only a negative. If V5
   won, V3 is an informative contrast (embedding vs mid-layer).
Skip V2 / V2b / V2c / V4 in the first pass — they are refinements. Reach for them only if V3 or
V5 shows signal and you want to understand which ingredient mattered (V4 = add multi-head;
V2/V2b/V2c = simpler embedding-layer combines). Both first-pass runs go to ~6500 iters with
checkpoints at 1500/3250/5500/6500, each vs its matched no-hub baseline (the existing baseline
from the alpha experiment may be reusable if the config matches).
Verdict: the existing embedding-additive result (baseline +0.0504 >= hub +0.046) is the number
to beat on anchor-layer Test B. If a variant's hub gap EXCEEDS its baseline gap (and the lead
grows over checkpoints), that is the architecture to carry forward — then ablate V3's upgrades
1/2/3 to see which mattered. If both V5 and V3 land ~baseline at 6500, architecture-only is a
controlled negative (longer will not flip it — tested); move to Section 2 (objective) or 3
(finetuning).

## 2. OBJECTIVE + ARCHITECTURE  [salvage route; adds translation data]

If architecture-only stays ~baseline, add explicit cross-lingual pressure. The anchors learn
per-language structure because nothing rewards bridging; these supply the missing reward. All
require SOME parallel data (spends the "no parallel data" advantage). Run WITH the matched
baseline and anchor-layer Test B. Optionally also run objective + OLD (embedding-additive)
architecture as a control, to isolate how much the objective alone contributes vs the
architecture.

Ordered strongest first:
- **2a. Sentence-pair translation LM (TLM, XLM-style)** — concatenate a translation-equivalent
  sentence pair, causal/masked LM over both, so one language's context predicts the other.
  Cross-lingual pressure arises NATURALLY through the LM loss, no explicit alignment term.
  Contextual, strongest option. Data: OPUS / CCMatrix.
- **2b. Contrastive sentence alignment** — pull pooled reps (or anchor distributions) of
  translation-equivalent sentences together, non-pairs apart (LaBSE-style). Explicit loss.
- **2c. Word-level alignment loss** — pull translation WORD pairs' anchor distributions (or
  hidden states) together using the existing LLM (GPT-4o) dictionary (4,804 tuples). Weaker
  (lexical only) but uses data already on hand. Easiest to add.
- **2d. Input code-switching (PREALIGN-style, input-only)** — swap some INPUT tokens for
  translations; keep the prediction TARGET in the original language (avoids mixed-language
  generation). Needs the dictionary.

---

## 3. FINETUNE A PRETRAINED MODEL  [highest-probability positive; held last]

In a pretrained multilingual model, "dog" ~= "cho" ALREADY exists, especially mid-network, so
the anchors only EXPLOIT existing alignment rather than CREATE it. Removes the root obstacle;
cheap (finetuning). Build the Section-1 upgraded block here (mid-layer + decoupled + gate) —
this is where it has the best chance, because the pretrained mid-layers are already aligned.

What to run:
- Base: pretrained multilingual model (e.g. pretrained Qwen3-0.6B), inject the upgraded block.
- Variants: (a) FREEZE base, train only hub — cleanest test of "exploit existing structure";
  (b) unfreeze base, small LR.
- Baseline (REQUIRED): same base finetuned the same way WITHOUT the hub.
- Metric: anchor-layer Test B vs the no-hub baseline.

---

## 4. MECHANISM refinements  [last; only after a positive result]
Optimize a working mechanism; do NOT create cross-lingual pressure.
- Top-k sparse selection ON THE ADDITIVE form (5-10 of 1000) — selection already reaches
  effective ~20 anchors, low expected value. (NOTE: this is distinct from V2c, which CONCATS the
  top-k anchor vectors — a real architecture change, in Section 1. Here it means merely sparsening
  the additive weighted sum, a refinement.)
- Fewer anchors (256/512) — Probe 1 showed ~45% of 1000 wasted; efficiency, not a fix.
- Fancier similarity than cosine — cosine is not the bottleneck.

---

## Do NOT
- Keep a fixed alpha alongside a learnable linear/gate (redundant — it absorbs any constant
  scale).
- Put architecture-only effort into a fancier EMBEDDING-layer combination and expect it to
  beat baseline — that is exactly where the base embeddings already win (redundant).
- Measure only input-embedding Test B for MID-layer anchors — probe at the anchor layer.
- Spend runs on mechanism refinements (4) while Test B is still flat.
- Expect "train longer" to change the verdict (tested).

## Always report
Every experiment runs WITH the matched no-hub baseline and the baseline-controlled Test B at
the ANCHOR LAYER (does the hub gap EXCEED the no-hub gap?). Test A alone is not sufficient — a
significant-but-tiny anchor-sharing gap does not imply functional alignment.
