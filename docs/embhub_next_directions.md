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
data, and beat the no-hub baseline. The primary variants are V2, V3, V4, V5, V6, V6f (V2 also has
cheap sub-variants V2b and V2c/+tail/+buckets). Run each as a separate experiment, each with its
own matched no-hub baseline. See "Suggested run order" at the end of this section for which to
run first — you do NOT need to run all of them. (NOTE: the "Shared elements" below apply to
V2-V5; V6 and V6f deviate where stated in their sections — no safe init [curriculum instead],
and renormalized top-k selection.)

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

ANCHOR COUNT — N = 128 FOR ALL NEW VARIANTS (not just V6f):
- N=1000 is empirically oversized: Probe 1 showed ~45% globally dead, ~550 effectively used —
  and a large N is what makes language partition AFFORDABLE in any variant. Small N applies the
  interference pressure everywhere; in V2-V5 (optional anchors) it costs nothing and prevents
  partition if the anchors are used; in V6/V6f it is a core mechanism.
- Default N = 128 (with k = 10, ~8% of the codebook per token — still sparse). Fallbacks are
  DIRECTIONAL: partition persists at 128 -> drop to 64 (but 64 makes k=10 select 16% of the
  codebook and inflates random-pair chance overlap, weakening Test B discrimination — use only
  if needed); concept-only PPL catastrophic at 128 (bottleneck too tight) -> raise to 256 (but
  256 lets 6 languages carve ~40 private anchors each — partition becomes affordable again).
- RECALIBRATE N-dependent diagnostics: max entropy = log(N) = 4.85 at N=128 (vs 6.91 at 1000),
  so the old "sharp selection ~ entropy < 4" rule of thumb is WRONG at N=128 (4 is near-uniform
  there) — read entropy relative to log(N). Top-k mass, dead-frac, and Jaccard baselines also
  shift with N. And at small N the frequency-matched RANDOM-pair control in Test B is
  load-bearing (chance overlap is higher), not a formality.

LEARNING RATE — CRITICAL (do not get this wrong):
- The 75x LR multiplier applies ONLY to the temperature parameter `log_logit_scale`. It exists
  because that single scalar has a tiny gradient and would otherwise stay frozen.
- Do NOT put the contribution parameters on 75x. `W` / `W_mix` (V2), `Linear_v`, `Linear_g`,
  `anchor_keys`, `anchor_values` (V3-V6f — this includes V6 and V6f's anchors) all stay on the
  NORMAL base LR (same as the rest of the model). In every variant the ONLY 75x parameter is
  `log_logit_scale`. Boosting them makes the anchor contribution grow FASTER than the model can adapt and
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
`output = Linear_out([x ; GELU(Linear_v(mixture))])`, with `Linear_v: d -> d` (so the concat
stays 2d and `Linear_out: 2d -> d` exactly as in V2). NOTE: putting a PLAIN linear on the
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
(NOT a contradiction with V6/V6f and the anchors-only pattern modes using RENORMALIZED weights:
different jobs. V2c's weighted slots sit ALONGSIDE x, so preserving absolute selection
confidence — Option B — adds information. A STANDALONE mixture must carry full magnitude on its
own, so it needs renormalization. Rule of thumb: raw weights when the mixture accompanies x;
renormalized when the mixture must stand alone.)

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

### Variant V5 — MID-LAYER placement  [the placement bet; the V3 (or V4) block on a mid layer]
Put the anchor block (the V3 block, or V4 if multi-head helped) after a MIDDLE transformer
layer (~layer 6-14 of 28; better: the layer chosen by test T5) instead of at the embedding.
`x` is then the mid-layer hidden state.
Rationale: at the embedding layer the anchors see only token identity (easiest signal =
language identity) and compete with embeddings that ALREADY hold the embedding-level
cross-lingual signal -> redundant -> ~baseline (the observed result). Mid-network is where
multilingual models form language-AGNOSTIC representations, so anchors there attend over
already-partially-aligned states AND can capture DEEPER alignment the base EMBEDDINGS do not
hold — i.e. NON-redundant structure the hub can actually add. (V5 is one of the two co-lead
bets: V5 = "the structure lives deeper", V6f = "the structure needs forcing".)
HONEST CAVEAT: the same redundancy constraint applies at the mid layer too — the BASELINE's
layer-10 states are also language-agnostic (that is exactly why alignment lives there), so the
hub must add alignment beyond what the baseline's mid-layer already develops on its own
(hence the per-layer verdict rule: beat the baseline's gap AT THAT LAYER, measured via T5).
V5 improves the odds (richer structure to latch onto; the decoupled value space lets anchors
encode something other than the hidden state) but does NOT remove the objective-level problem:
nothing in the LM loss rewards cross-lingual sharing. Contribution is added to the hidden
state at that layer; rest of the model unchanged. Optionally place blocks at multiple depths.
Safe init for V5: same as the block it uses (V3/V4 init: `Linear_v` ~0 or `anchor_values` ~0,
`Linear_g` bias strongly negative) so it starts `output ~= x`.

### Variant V6 — separate anchor path with stochastic replacement (the original idea)

THE IDEA, plainly: V2-V5 all attach the anchors to the token embedding as an OPTIONAL extra —
additive, gated, or concatenated. They are NOT yet tested, but they may share the limitation
the OLD additive architecture demonstrably had (the controlled negative): the token's own
private embedding already carries everything the LM loss needs, so the model can satisfy the
loss while ignoring the anchors. V6 attacks that possible limitation directly: instead of
always combining anchors WITH the embedding, make the anchors sometimes REPLACE the embedding
entirely. If, for some tokens, the representation IS the anchor mixture and nothing else, then
the anchors MUST carry real predictive content — they cannot be decorative. Necessity by
construction, no new loss term, no special data.

(A "replace everywhere" extreme — anchors-only for ALL tokens — is deliberately NOT included:
from scratch, the embeddings would then train only through the weak retrieval-weight gradient,
so the addressing may never become good (chicken-and-egg), and fixing that would require
training a second embedding-only model, which is expensive and disconnects the two models'
layers. The stochastic mix below keeps the embedding trained on the plain path while still
forcing the anchors to stand alone part of the time.)

**V6-mix: stochastic per-token replacement — 50/40/10.**
Per token, per training step, roll a die:
- ~50%: use the plain token embedding (baseline path — keeps LM quality anchored),
- ~40%: use embedding combined with the anchor mixture (combine form: see note below),
- ~10%: use the ANCHOR MIXTURE ONLY (embedding serves only as the retrieval address).
The 10% anchors-only slice is the teeth: for those tokens the model must predict from the
anchors alone, so the anchors must genuinely carry meaning — while the 50% plain slice keeps
overall LM quality. (Same trick family as BERT's 80/10/10 masking or modality dropout.) At
inference, use the combined form; comparing inference modes is itself informative.

```python
# V6-mix forward (training; use the N=128 default like all variants. V6-mix supplies NECESSITY;
# V6f ADDS scarcity's second role + the capped residual. Large N here would re-allow partition.)
# tok_emb: (B, T, d)
w = softmax(cos(tok_emb, anchor_keys) * log_scale.exp().clamp(max=100))  # (B, T, N)
topw, topi = w.topk(k)                               # k ~= 10
w_norm = topw / topw.sum(-1, keepdim=True)           # renormalized (mixture must stand alone)
mixture = (w_norm.unsqueeze(-1) * anchor_values[topi]).sum(-2)   # (B, T, d)

# mode sampling MUST be PER TOKEN — a (B, T) tensor, NOT one rand() per batch/sequence.
# (One draw per sequence would send whole sequences anchors-only: a different, wrong design.)
r = torch.rand(B, T, 1, device=tok_emb.device)       # p_only/p_both from the curriculum below
out = torch.where(r < p_only,          mixture,                  # anchors only — the necessity
      torch.where(r < p_only + p_both, tok_emb + mixture,        # plain ADD (see note)
                                       tok_emb))                 # plain path
```

WHICH COMBINE FORM for the "both" mode (V6 can in principle wrap any of V2-V5's combines):
use PLAIN ADD (`tok_emb + mixture`). (This is V1's combination OPERATOR but not V1: there is
NO alpha — the mixture must stand at full strength for the anchors-only mode — and retrieval is
renormalized top-k over decoupled keys/values, not V1's full softmax with keys=values.)
Reason for plain add — MODE CONSISTENCY: in the 10% anchors-only mode
the mixture feeds the transformer DIRECTLY, so if the 40% mode passed it through a learned
combiner (V2's concat+linear) or a gate (V3), the mixture would mean different things in
different modes (raw object in one, transformed in another), which makes the anchors' job
incoherent. Plain add keeps `mixture` the same object in every mode. It also adds zero
parameters, and V3's gate would be counterproductive here (a gate that can close the anchor
path works against the necessity pattern). The renormalized top-k weighting already ensures the
mixture has standalone magnitude. Keep V6 at the EMBEDDING layer; mid-layer placement stays
V5's separate bet (replacing mid-layer hidden states with anchors is far more disruptive).

NO SAFE INIT — CURRICULUM instead (same reason as V6f): `output ~= x` is impossible when the
output must sometimes BE the mixture, and at step 0 the anchors are random noise. Start at
100/0/0 (all plain) and anneal to 50/40/10 over the first few thousand steps, so the anchors
are shaped while optional and the necessity pressure turns on only once they carry something.
Concretely: `ramp = min(1, step / anneal_steps)` (anneal_steps ~ 2000-3000);
`p_only = 0.10 * ramp`, `p_both = 0.40 * ramp` (plain-path probability is the remainder).
NOTE: anchors (and the temperature) receive gradient only on the ~50% of tokens in the
anchor-using modes once annealed — expect the temperature ramp and anchor training to be
somewhat slower per step than in always-on variants; that is normal, not a bug.

What V6 does and does NOT fix:
- It fixes IGNORABILITY (the anchors become load-bearing — the limitation V2-V5 MAY inherit
  from the tested additive design).
- It does NOT by itself fix PARTITIONING at LARGE N: necessity forces the anchors to be used,
  but if N were large each language could still claim its own private (now load-bearing) set —
  real content, but per-language. Two things address this: the N=128 default (shared elements)
  removes most of the room to partition, and V6f adds the capped residual so legitimate
  per-language content has a home while the scarce concept pool is pushed to share.
That second gap is exactly what V6f adds.

VERDICT and MONITORING for V6 (same three dials as V6f). Dials 1-2 are evaluated in the
deterministic INFERENCE mode (`tok_emb + mixture` on ALL tokens — no stochastic sampling at
eval): (1) anchor-layer Test B vs matched baseline; (2) per-language PPL vs baseline. Dial (3)
anchors-only PPL: force the mixture-only mode on ALL tokens — if catastrophic, the anchors are
not actually load-bearing. (Init note: `anchor_keys`/`anchor_values` init at embedding scale,
std ~0.02, like the rest — the mixture must live at the embedding's magnitude since it adds to
and sometimes replaces it.) Because V6 uses HARD
top-k routing, the V2c routing caveat applies: watch `dead_anchor_frac` over the first ~1000+
steps (rich-get-richer anchor death; with load-bearing anchors this is a real risk — at the
N=128 default it is milder than at large N, but if severe, reduce N further or add a V2c-style
aggregated tail).

THE V6 PATTERN ON V2-V5: EVERY variant can carry the pattern. The split is WHICH FORM:
- FULL pattern (token-only + anchor-only + mix): V2, V2b, V3, V4 — and V5 with a known cost.
- LITE pattern (token-only + mix, no anchors-only): available to ALL variants; the ONLY form
  V2c can carry (no faithful anchors-only mode exists for it — see recipes).
Analysis: coherent for V2-V4 via one fix, but every coherent full hybrid CONVERGES TO V6 — so
for a NECESSITY experiment, run V6 pure; the hybrids are post-signal ablations.
- The fix that makes it coherent (for V2/V2b/V3/V4): anchors-only mode bypasses the variant's
  COMBINE step ONLY (V2's `Linear([x;.])`, V3's gate) — content TRANSFORMS (V2b's MLP, V3's
  `Linear_v`) are kept and applied identically in BOTH modes, so the mixture is ONE object
  everywhere. This works because the anchors-only constraint ("be a full standalone
  representation") is the STRONGEST constraint and subsumes the combiner-input role.
- But trace the training dynamics: once anchors-only pins the mixture to standalone
  embedding-space semantics, the combiner in the mix mode either (a) learns ~pass-through
  (rediscovering plain add with extra parameters) or (b) learns to SUPPRESS the mixture — an
  escape hatch that shrinks necessity from 50% of tokens (V6's forced full-strength add) to
  only the 10% anchors-only slice. So V2/V3/V4 + pattern = V6 with extra parameters and
  WEAKER necessity. The pure plain-add V6 is the strongest form of the pattern.
- V5 + pattern: coherent in mechanism (retrieval at layer L is contextual, so anchors WOULD be
  trained to imitate hidden states — same learning loop as at the embedding), but with a
  fundamentally worse target. At the embedding the mixture must only reproduce TOKEN IDENTITY
  (~17 bits, fully learnable -> replacement damage trains to ~zero). At layer L it must
  reproduce the ACCUMULATED PREFIX CONTEXT (arbitrary specifics — names, numbers, references —
  that a k-sparse mixture over a fixed codebook cannot carry) -> a PERMANENT damage floor on
  every replaced position, and the anchors' job shifts from "cross-lingual token concepts" to
  "coarse context summarizers". This is effectively a VQ-style mid-network bottleneck — a
  legitimate but DIFFERENT (and costlier) experiment that would also confound V5's placement
  bet with V6's necessity bet. Keep out of the first pass.
- Post-signal upgrades to V6 (only after V6/V6f shows signal): a MODE-CONSISTENT transform
  (`mixture' = Linear_v(mixture)` applied identically in EVERY mode, ordinary init — the
  curriculum handles early safety), and/or V3-style extras on the mix mode, understood as
  ablations of "does expressiveness help a working necessity mechanism?".
MODE IMPLEMENTATION per variant (if a hybrid is ever built — exact recipes):
- Token-only mode = BYPASS THE BLOCK ENTIRELY: output the raw `x` (embedding, or hidden state
  for V5) as if the hub did not exist. TRAP: for V2, do NOT implement token-only as
  `Linear([x ; 0])` — that equals `x` only at init (`[I|0]`) and silently diverges from the
  plain path as `W_x` trains away from identity. Token-only must be the true baseline path.
  (For V3/V4 this is natural — the block is residual, so token-only = skip the `gate*update`
  branch; for V2 the bypass must be explicit because the block is not residual.)
- Anchor-only mode = the RAW retrieved mixture, bypassing every combiner.
  WARNING (applies to V2/V2b/V3/V4): do NOT use the FULL-softmax mixture in anchor-only mode.
  Near-uniform softmax (early training / early curriculum) makes `softmax @ anchors` ~= the
  MEAN of all anchors — the SAME vector for every replaced token: the V1 born-dead failure
  reproduced inside a mode. Fix: any full-pattern hybrid should switch its selection to TOP-K
  RENORMALIZED throughout (BOTH modes, one mixture object — mode-consistent), which keeps
  mixtures token-differentiated even when weights are soft (the top-k SET differs per token).
  This is how V6 does it, and yes — it moves the hybrid even closer to V6.
  V2: top-k renormalized `w~ @ anchors` (keys = values = anchors; with the switch above).
  V2b: do NOT bypass the MLP — that would make the mixture a different object per mode. Use the
  MODE-CONSISTENT form: compute `mixture' = GELU(Linear_v(w~ @ anchors))` ONCE and use it in
  BOTH modes (anchor-only = `mixture'`; mix = `Linear_out([x ; mixture'])`). `Linear_v` gets
  ORDINARY init (the curriculum protects the anchor-only mode); `Linear_out` keeps its `[I|0]`
  safe init (protects the mix mode).
  V2c family: NO FAITHFUL anchors-only mode exists. Collapsing to a weighted sum abandons the
  slot structure that defines V2c (you would be running V2's mode, not V2c's); feeding the
  slots through the combiner with a zeroed x-block is the V2 trap; and a separate slots-only
  projection reintroduces mode inconsistency. V2c can only carry PATTERN-LITE (see below).
  V3/V4: MODE-CONSISTENT form (same doctrine as V2b): compute the shared object
  `update = Linear_v(w~ @ anchor_values)` once; anchor-only = `update`; mix = `x + gate*update`.
  Only the GATE (the combine step) is bypassed in anchor-only. Init split for the hybrid:
  `Linear_v` switches to ORDINARY init (the curriculum protects anchor-only — a zero-init
  `Linear_v` would emit ~zero standalone representations for thousands of steps); the gate
  KEEPS its bias ~ -5 safe init (protects the mix mode).
  V5: the V3-block's shared object (`Linear_v(w~ @ anchor_values)`, per the V3/V4 recipe)
  replaces the layer-L hidden state (the costly case analyzed above).
- Mix mode = the variant's normal forward — with ONE change in full-pattern hybrids: the
  selection is the switched TOP-K RENORMALIZED version in this mode too (both modes share one
  mixture object; "unchanged" refers to the combine step, not the selection).
- All hybrids need the V6 CURRICULUM on the mode probabilities (full: 100/0/0 -> 50/40/10;
  lite: 100/0 -> 50/50) regardless
  of the variant's safe init — safe init protects the MIX mode's combiner; the curriculum
  protects the ANCHOR-ONLY mode (anchors are random noise at step 0 in every variant).

Summary table (modes per variant; "raw x" always means bypass the block entirely):

| variant | pattern forms available | token-only | anchor-only | mix |
|---|---|---|---|---|
| V2 | FULL or LITE | raw x (NOT `Linear([x;0])`) | top-k renorm `w~ @ anchors` | normal forward |
| V2b | FULL or LITE | raw x | `GELU(Linear_v(w~ @ anchors))` — MLP in BOTH modes | normal forward |
| V2c    | LITE only | raw x | — (no faithful mode) | normal forward |
| V3/V4  | FULL or LITE | x (skip the `gate*update` branch) | `Linear_v(w~ @ anchor_values)` — transform in BOTH modes, gate bypassed | normal forward |
| V5     | LITE recommended; FULL possible but costly | h (hidden state, untouched) | replace h with `Linear_v(w~ @ values)` (permanent damage floor — see analysis) | normal forward |
| V6/V6f | FULL (native) | tok_emb | mixture / concept (native) | plain add / concept+resid |

TWO PATTERN OPTIONS any V2-V5 hybrid could use (recorded as available versions):
- FULL pattern ("Vx+pattern"): token-only + anchor-only + mix. Available to V2/V2b/V3/V4
  (via the raw-mixture fix + top-k renorm switch) and to V5 (possible, but with the permanent
  damage floor — a VQ-style bottleneck, a different and costlier experiment). NOT available to
  V2c (no faithful anchors-only mode). The analysis above applies: full hybrids converge to V6.
- REDUCED pattern ("Vx+pattern-lite"): token-only + mix, NO anchors-only mode. Available to
  EVERY variant (V2, V2b, V2c, V3, V4, V5) and SAFE everywhere — including V5, since the hidden
  state is never replaced (no context loss, no damage floor). For V2c it is the ONLY form. BUT be honest about what it is: without the anchors-only mode it has NO
  necessity mechanism. Randomly dropping the hub branch is BRANCH DROPOUT / stochastic depth —
  it trains the model to function WITHOUT the hub (robustness to the branch's absence), which
  is the OPPOSITE of forcing reliance on it. The anchors-only mode IS the teeth of the V6
  pattern; remove it and the pattern becomes a mild regularizer, not a necessity device. So
  pattern-lite resolves the V5 safety concern at the price of the pattern's entire point.
  Recorded as an option; NOT expected to change any variant's cross-lingual outcome.
  V2c+pattern-lite recipe (the one variant restricted to lite): token-only = raw `x`, bypassing
  the block entirely (the V2 bypass trap applies — never `Linear([x ; 0...])`); mix = V2c's
  normal forward `Linear([x ; w1*a_i1 ; ... ; wk*a_ik])`; no anchors-only mode. Curriculum on
  the mix probability (100/0 -> 50/50 over the first few thousand steps). Same caveat as all
  pattern-lite: this is branch dropout, not a necessity mechanism.

Strategic rule: keep first-pass variants PURE — V5 and V6f test two competing hypotheses, and a
chimera changes everything at once, making any result unattributable. Hybridize only after
something wins.

### Variant V6f — V6 + scarcity + capped residual (factorized concept/residual)  [co-lead with V5]

THE IDEA: V6f = V6-mix PLUS the two additions that close V6's remaining gap (load-bearing
anchors can still PARTITION by language when N is large). It factorizes every token into
`shared concept + small private residual`:
- Make the anchors a SMALL shared "concept codebook" (N ~= 64-128, not 1000): all 152k tokens,
  all 6 languages, must express their MEANING as a combination of the same few concept vectors.
  Small N means languages CANNOT claim disjoint anchor sets (with N=1000 they could, and did —
  Probe 1's partition, en-zh Jaccard 0.16). Every anchor is reused by all languages, so every
  anchor's gradient mixes all languages' demands -> language-neutral content is the low-conflict
  solution the optimizer prefers. Sharing by scarcity, not by a loss term. (Capacity is NOT the
  issue — C(64,10) combinations with continuous weights is astronomically more than 152k tokens
  need; what small N forces is a shared BASIS, which is the thing that must align.)
- Keep a small PRIVATE residual per token — the token's own embedding, NORM-CAPPED to at most
  ~30% of the concept's length — so genuinely per-language content (cultural terms, orthography)
  has a structural home (the balance point from the "don't erase language identity" concern),
  but the residual is too small to carry the whole meaning and bypass the codebook.
- Make the codebook LOAD-BEARING: during training, per token, randomly use concept-only ~10% of
  the time (the model must predict from the concept alone — the teeth), concept+residual ~40%,
  plain embedding ~50% (keeps LM quality anchored; same trick family as BERT's 80/10/10).

Formula: `repr(token) = concept + residual`, where
`concept = sum_j w~_j * anchor_v[i_j]` (renormalized top-k over the small codebook) and
`residual = clip(token_emb, norm <= 0.3 * ||concept||)`.

```python
class V6Factorized(nn.Module):
    def __init__(self, d=1024, N=128, k=10, r_budget=0.3):
        self.anchors_k = nn.Parameter(randn(N, d))   # keys (selection)
        self.anchors_v = nn.Parameter(randn(N, d))   # values (content)
        self.log_scale = nn.Parameter(log(14))       # learnable temp (75x LR, no WD — as usual)
        self.r_budget  = r_budget

    def forward(self, tok_emb, mode):
        # concept: retrieve from the small shared codebook
        q = normalize(tok_emb); K = normalize(self.anchors_k)
        w = softmax((q @ K.T) * self.log_scale.exp().clamp(max=100))    # (B,T,N)
        topw, topi = w.topk(self.k)                                     # top-k of N
        w_norm = topw / topw.sum(-1, keepdim=True)                      # RENORMALIZED (see note)
        concept = (w_norm.unsqueeze(-1) * self.anchors_v[topi]).sum(-2) # (B,T,d)

        # residual: private per-token part, norm-capped PER TOKEN
        max_norm = self.r_budget * concept.norm(dim=-1, keepdim=True)
        scale_dn = (max_norm / tok_emb.norm(dim=-1, keepdim=True)).clamp(max=1.0)
        resid = tok_emb * scale_dn        # shrink if too long, never stretch (min(1, ...))

        # stochastic necessity (per token, TRAINING only; inference uses concept + resid)
        if mode == "concept_only": return concept          # ~10%
        if mode == "both":         return concept + resid  # ~40%
        return tok_emb                                     # ~50% plain-embedding path
```

Notes that differ from V2-V5:
- Weighting is RENORMALIZED top-k (Option A) here, unlike V2c where raw softmax (Option B) was
  preferred: in concept-only mode the concept must stand alone as the FULL representation, so
  its magnitude cannot be allowed to shrink with selection confidence.
- `.norm()` = Euclidean vector length; the cap means "the private vector may be at most 30% as
  long as the concept — shrink it to that if longer, leave it if shorter." Relative (0.3 x
  concept's length) rather than absolute, so it stays correct as anchor scale grows in training.
  Compute per token (`dim=-1, keepdim=True`) — not one norm over the batch.
- NO SAFE INIT — a CURRICULUM replaces it. `output ~= x` is impossible when the output must
  sometimes BE the concept. At step 0 anchors are random noise, so concept-only tokens would get
  garbage. Instead: start at 100/0/0 (all plain-embedding) and anneal to 50/40/10 over the first
  few thousand steps — the codebook gets shaped while optional, then the necessity pressure
  turns on. The mode percentages and anneal length are dials.
- Dials to sweep: N (start 128; 64 if partition-like behavior persists), k (10), r_budget
  (0.3), mode mix (50/40/10), anneal steps.

VERDICT for V6f — three dials, not one (V6f WILL move PPL, unlike ignorable variants):
1. Anchor-layer Test B vs the matched baseline (are concept selections cross-lingual?). Keep the
   frequency-matched random-pair control — with a small codebook, chance overlap is higher.
2. Per-language PPL vs baseline (did the squeeze hurt the languages? — the balance check).
3. Concept-only PPL (evaluate with mode=concept_only): if catastrophic, the codebook is not
   actually load-bearing despite the 10% (the residual/plain modes are doing all the work).

Honest calibration: this is the strongest architecture-only design — it is the first that
reproduces the mechanism by which the shared transformer BODY demonstrably aligns (one scarce
substrate all languages are forced through), instead of offering an optional side-branch. Still
not guaranteed: the codebook can organize by frequency/syntax instead of translation-level
meaning, and the LM loss still has no explicit cross-lingual term. But its failure would be
genuinely informative: if scarcity + necessity + isomorphic data do not produce shared
semantics, nothing architecture-only will — move to Sections 2/3.

### MEASUREMENT (applies to EVERY variant)
If anchors live at layer L, their effect is in the LAYER-L hidden states, not the input
embeddings. So run the Test-B-style comparison on layer-L representations: do translation
pairs have more similar layer-L states WITH the hub than the no-hub baseline at layer L? For
V2-V4 (embedding layer) L = the embedding output; for V5, L = the mid-layer the block sits on.
Measuring only input-embedding Test B for a MID-layer block (V5) would MISS its effect. Keep
Test A (anchor overlap) as a secondary signal; the verdict is anchor-layer Test B vs the
matched no-hub baseline.

### ADDITIONAL TESTS (beyond Test B)

Test B only asks "are translation pairs geometrically close?". These add the questions it cannot
answer — does a task transfer, does the model treat translations as interchangeable, does the
model's OUTPUT behave cross-lingually. All are forward-pass or linear-probe cheap (no training).

Each test below: the QUESTION it answers, HOW to run it (numbered), what a WIN looks like, and
what to WATCH OUT for. One rule shared by T3/T6/T7: always include the frequency-matched RANDOM
control, or you measure "likes common / same-language words" instead of translation.

---

**T1 — Transfer probe.** (all variants; hub vs baseline)
QUESTION: does a task learned in English carry to other languages? (This is the project's actual
claim.)
HOW:
1. Freeze both models (hub, baseline).
2. Represent each sentence = mean-pool the hidden states at a chosen layer. For V6/V6f use the
   inference-mode representation (`tok_emb + mixture` / `concept + resid`).
3. Train a LINEAR classifier on English XNLI on top of that representation (pair features
   `[u; v; |u-v|; u*v]`; 20-50k examples is enough for a linear head; early-stop on English dev).
4. Test it UNCHANGED (zero-shot) on the other languages. Repeat with 3-5 seeds, report mean+-std.
5. Do this at the SAME layer depth in both models (fair comparison: the hub is just part of the
   hub-model's computation up to that depth).
WIN: hub's zero-shot foreign accuracy > baseline's, beyond seed noise.
WATCH OUT: at the EMBEDDING layer, mean-pooling is bag-of-words (no word order), so XNLI is
near-chance for BOTH models and the delta is unmeasurable. Read T1 at DEEPER layers (mid, last),
which have order and rise above chance. (For V5 the anchor layer is already deep, so it works
there.) Probe several layers in one forward pass — nearly free.

---

**T2 — Language-decodability.** (hub-internal diagnostic; no baseline)
QUESTION: do the anchors just encode "which language is this?" (the partition failure)?
HOW:
1. Sample equal tokens per language from the eval set.
2. Input = the token's full N-dim anchor-weight vector `w`.
3. Train logistic regression to predict the token's language; report accuracy vs 6-way chance
   (16.7%). Track across checkpoints.
WIN (the intended direction): decodability FALLING while Test B RISES = shared structure
replacing language partition.
WATCH OUT: target is NOT zero — some language-specific content is legitimate (cultural,
orthographic). And loanwords (identical token in several languages) have identical `w` but
different labels, so there is an error floor — read the TREND, not the absolute number.

---

**T3 — Mixture-interchange.** (embedding-layer variants: V2-V4, V6, V6f; hub vs its own swaps)
QUESTION: does the model treat "dog" and "cho" as interchangeable — causal proof the anchors
carry shared meaning?
HOW (single-token pairs only):
1. Precompute each word's anchor mixture (context-free at the embedding layer — one lookup).
2. Take eval sentences containing the foreign word `w_x`, keeping only ones with >=10 tokens
   AFTER it. Run normally; record the summed log-prob of ONLY those following tokens -> `L_clean`.
3. Re-run with `w_x`'s mixture replaced by its English translation's mixture (leave `w_x`'s own
   embedding/residual untouched — swap ONLY the anchor pathway); record `L_swap`.
   Damage_translation = L_clean - L_swap.
4. Control: same swap but with a FREQUENCY-MATCHED RANDOM English word's mixture ->
   Damage_random.
5. Over many pairs (paired stats): compare Damage_translation vs Damage_random.
WIN: translation-swap hurts MUCH LESS than random-swap = the model barely notices when you
substitute the real translation's concept = anchors encode shared meaning.
WATCH OUT: measure ONLY the tokens AFTER the swap. Under causal masking, tokens before it and
its own prediction are identical in both runs; including them dilutes the signal to zero. If both
swaps hurt ~zero, the anchor pathway is not load-bearing (cross-check the anchors-only PPL dial).
NOT for V5 (mid-layer mixtures are contextual — cannot be precomputed or swapped across contexts).

---

**T4 — Contribution share.** (V3/V4/V5; rough proxy for V2)
QUESTION: is the model actually USING the hub, or ignoring it (the ignorability failure)?
HOW: over the eval set, log per token (aggregate per language) the effective contribution share
`||gate * update|| / ||x||` (how big the hub's contribution is vs the token). Also log mean gate
value, but the norm ratio is the decisive dial (a gate can be open while the update is tiny).
- V3/V4/V5: clean, because the block is residual (`x + gate*update`).
- V2: non-residual, so use `||W_mix @ mixture|| / ||output||` as a ROUGH proxy (not identical;
  read as a trend, since `W_x` drifts from identity over training).
WIN / READ: share ~0 late in training = hub declined (ignorability confirmed, mid-run, without
waiting for Test B). Share healthy for some languages only = hub used as a language patch, not a
bridge.

---

**T5 — Layer-sweep (for V5, BEFORE training).** (runs on EXISTING baseline checkpoints)
QUESTION: which layer should V5's anchors sit at?
HOW:
1. On a trained baseline checkpoint (the alpha-experiment baseline if you still have it;
   otherwise any monolingual-trained no-hub baseline — no NEW training needed), for each layer L:
   take single-token
   translation pairs, mean-pool each word's hidden states over its IN-CONTEXT occurrences
   (deeper layers are contextual — use real occurrences, not isolated tokens).
2. Compute the translation-vs-frequency-matched-random similarity gap at every L. One forward
   pass with all layer outputs retained.
WIN / USE: the layer where the gap PEAKS is where cross-lingual structure naturally lives — put
V5's block there instead of guessing "~layer 10". Bonus: these per-layer gaps ARE the baseline
numbers V5's verdict must beat.
WATCH OUT: raw cosines grow large at deep layers (hidden states are anisotropic — everything is
similar to everything); that is why the GAP (translation minus random) is the reading, since
both sides inflate equally.

---

**T6 — Translation retrieval (BLI).** (all variants; hub vs baseline)
QUESTION: can you FIND a word's translation as its nearest neighbor? (The cleanest, cheapest
cross-lingual number — no probe training, no data to build.)
HOW (single-token pairs):
1. Get each word's representation at the tested layer (inference-mode for V6/V6f).
2. Build a candidate pool = all single-token target-language words in the pair set.
3. For each English word, score every candidate by CSLS = `2*cos(e,x) - r(e) - r(x)`, where
   `r(e)` = e's average cosine to its ~10 nearest TARGET-language candidates, and `r(x)` = x's
   average cosine to its ~10 nearest SOURCE (English) words (the two densities point in OPPOSITE
   directions — this is what corrects hubness). Use CSLS, NOT raw cosine, which inflates BLI.
   (CSLS is from Conneau et al., "Word Translation Without Parallel Data" — arXiv 1710.04087,
   2017 preprint / ICLR 2018, the SAME MUSE paper and dictionary this project already uses. It
   is the field-standard BLI nearest-neighbor retrieval metric. Formula direction confirmed
   against the published equation: for source word e and target candidate x, the penalty r(e)
   is e's mean cosine to its TARGET-side neighbors and r(x) is x's mean cosine to its SOURCE-side
   neighbors. Note: standard but not SOTA — later methods (unsupervised-MT BLI 2019, cross-encoder
   rerankers) beat it by a few points; fine for a cheap diagnostic, reconsider only if BLI becomes
   a headline number.)
4. Record whether the true translation is rank-1 (P@1) / in the top-5 (P@5). Report per language
   pair, hub vs baseline at the same layer.
WIN: hub's P@1/P@5 > baseline's. (Stronger than Test B: there the translation only had to beat
random pairs; here it must beat EVERY other word.)
WATCH OUT: absolute P@1 is low at 0.6B, especially en-zh — read the hub-minus-baseline delta.

---

**T7 — Generative behavior (code-switch continuation).** (all variants; hub vs baseline)
QUESTION: does the MODEL'S OWN OUTPUT (through the LM head) behave more cross-lingually with the
hub? (Closest thing to a real translation-task result; similar to PREALIGN's CLKA.)
HOW:
1. Build minimal prompts whose correct next token is a known translation, from the word
   dictionary — e.g. "dog in Vietnamese is ___" (target `cho`), or "cho = ___" (target `dog`).
2. For each prompt, read `log P(correct translation token | prompt)` — one forward pass.
3. CONTROL: same prompt, `log P(frequency-matched random target-language word)`.
4. Metric = mean of `[log P(correct) - log P(random)]`. Compare hub vs baseline.
WIN: hub's margin > baseline's = the hub makes the model GENERATE correct cross-lingual
continuations more strongly.
WATCH OUT: the random control (same prompt) is essential — it cancels the prompt's oddness and
isolates the translation-specific effect. Dictionary is used only to BUILD prompts, never to
train.
(REJECTED alternative: "prepend the English sentence, measure PPL drop on its translation." The
model never saw en-then-translation in training, so that input is out-of-distribution and a null
would be uninterpretable; it also lacks a within-prompt control. Do not use it.)

---

**T8 — MEXA (all variants; hub vs baseline; standardized sentence-level alignment).**
PRIMARY INSTRUCTION — USE THE OFFICIAL IMPLEMENTATION: https://github.com/cisnlp/MEXA
(Apache-2.0; verified as the paper authors' code — Kargaran et al., arXiv 2410.05873, ACL
Findings 2025). Run their `embed_extractor.py` (default `--embedding_type embd_weighted`) then
`compute_mexa.py` (pivot `eng_Latn`) on BOTH the hub model and the matched baseline, and compare
the scores per language pair. Follow the repo's README as the source of truth; the explanation
below is only to help you understand what those scripts do and why — if it ever conflicts with
the repo, the REPO wins.
QUESTION: how well do parallel sentences align at each layer, on a robust metric that (a)
correlates ~0.90 with real downstream multilingual performance and (b) is comparable to
published numbers?
WHY it is better than a plain cosine average: it does NOT use absolute cosine values (those
suffer anisotropy/hubness — non-parallel sentences can score as high as parallel ones). Instead
it is a binary RETRIEVAL check per sentence pair.
HOW:
1. Take ~100 parallel sentences per language pair (En + XX). FLORES-200 devtest is the standard
   source; 100 is enough (the paper shows chance of a high score is ~0.0002 at n=100).
2. Sentence embedding at layer l = TOKEN-POSITION-WEIGHTED average of that layer's hidden states:
   `e_l = sum_t w_t * h_lt` with `w_t = t / (sum_k k)` (later tokens weighted more — corrects the
   causal-attention bias where late tokens carry more context). This weighted-average + later
   mean-pooling over layers is the paper's best-correlating setting.
3. For layer l, build the n x n cosine matrix C where `c_ij` = cos(En sentence i, XX sentence j).
   Diagonal `c_ii` = true parallel pairs.
4. MEXA score at layer l = fraction of i where `c_ii` is STRICTLY GREATER than every off-diagonal
   in BOTH its row and its column:
   `mu_l = (1/n) * sum_i  1[ c_ii > max_{j != i} { c_ij , c_ji } ]`.
   (This is P@1 sentence retrieval in both directions at once — hence robust to hubness, since
   only the RANK of the true pair matters, not the absolute similarity.)
5. Combine layers by MEAN pooling of `mu_l` (paper default) to get one score per language pair.
WIN: hub's MEXA > baseline's (per language pair, same layers). Because MEXA correlates ~0.90
with downstream tasks, a hub gain here is strong evidence of real multilingual gain, and the
number is comparable across models/runs using the same sentences + setting.
WATCH OUT: needs ~100 PARALLEL SENTENCES per pair (FLORES-200 devtest, first 100 lines — has
vi/zh/ru/de/ar) — used only for EVALUATION, never training. Report per language pair. Cheap: one
forward pass over ~100 short sentences per language, no probe training.
DATA FORMAT for the repo scripts: one text file per language (e.g. `eng_Latn.txt`,
`vie_Latn.txt`), line i parallel across files; FLORES-200 devtest first 100 lines. (See the
PRIMARY INSTRUCTION above — run the official scripts, do not reimplement the binary-retrieval
logic.)

HOW THEY FIT (weakest to strongest cross-lingual claim): T4 "hub is used" -> T2 "not just
language tags" -> Test B "translations geometrically close" -> T6 "close enough to be nearest
neighbors" -> T3 "interchangeable = same meaning" -> T1 "a task transfers" -> T7 "the model
itself behaves cross-lingually". T5 is a setup tool for V5, not a verdict.

REPORT related vs distant pairs SEPARATELY: en-de / en-vi (Latin script, loanword-heavy) apart
from en-zh / en-ar / en-ru (distant, other scripts). Architecture-only alignment is strong for
related pairs, weak for distant ones — a single average hides the pattern; en-zh is the stress
case.

Priority if time-constrained: T6 + T8 (MEXA) + Test B every checkpoint (all cheap, and MEXA is
the one that correlates with real downstream performance and is comparable to published numbers);
T2 + T4 as always-on diagnostics; T1 + T3 + T7 on checkpoints that already look promising; T5
once before any V5 run.

Suggested run order for step 1 (do NOT run everything — this is a two-run first pass):
1. RUN V5 and V6f as the two first-pass runs — they test the two DIFFERENT hypotheses about why
   everything failed: V5 = "the structure lives deeper" (placement), V6f = "the structure needs
   forcing" (scarcity + necessity). Both are higher-value than any embedding-layer combine.
   (Finetuning, Section 3, remains higher-probability overall but is deliberately held last.)
2. V3 (at the embedding layer) only as optional confirmation if both lose — it tells you the
   embedding layer is also dead (expected) before declaring architecture-only a negative.
Skip V2 / V2b / V2c / V4 / V6 in the first pass — they are refinements or ablations. Reach for
them only if something shows signal and you want to understand which ingredient mattered (V4 =
add multi-head; V2/V2b/V2c = simpler embedding-layer combines; V6 = V6f minus scarcity and the
residual cap — the ablation that attributes V6f's result to necessity vs scarcity). First-pass runs go to ~6500 iters
with checkpoints at 1500/3250/5500/6500, each vs its matched no-hub baseline (the existing
baseline from the alpha experiment may be reusable if the config matches; V6f additionally needs
its three-dial verdict — see its section).
Verdict rule: a variant wins if its hub gap EXCEEDS its matched baseline's gap AT THE SAME
LAYER (and the lead grows over checkpoints). Note the reference number is PER-LAYER: for V3
(embedding) the baseline reference is the known +0.0504; for V5 (layer ~10) the baseline's
layer-10 gap does not exist yet and must be MEASURED from the baseline run — do not compare
V5's layer-10 gap against the embedding-level +0.0504. If a variant wins, carry it forward and
ablate V3's upgrades 1/2/3 to see which mattered. If both V5 and V3 land ~baseline at 6500,
architecture-only is a controlled negative (longer will not flip it — tested); move to Section 2
(objective) or 3 (finetuning).

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
- Top-k sparse selection ON THE ADDITIVE form (5-10 of N) — selection already reaches
  effective ~20 anchors, low expected value. (NOTE: this is distinct from V2c, which CONCATS the
  top-k anchor vectors — a real architecture change, in Section 1. Here it means merely sparsening
  the additive weighted sum, a refinement.)
- Fewer anchors as a STANDALONE tweak — largely MOOT now: N=128 is already the default for all
  variants (shared elements). Sweeping N further (64 vs 128 vs 256) is a dial there, not a
  separate refinement.
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
