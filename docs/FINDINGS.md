# Warpweft — implementation log and findings

A running record of what we built, every training run and why we ran it,
what the measurements showed, and the mistakes worth remembering. Written
for the version of me that comes back to this in six months.

---

## The goal

Learn how the transformer architecture actually works by building a
decoder-only language model from scratch in Elixir — no framework model
graph, parameters visible as a plain nested map, every architectural
choice a flag we can measure rather than a decision we take on faith.

Secondary goals that shaped the design:

- **Measure, don't assert.** Every claim about architecture or performance
  in this repo should have a number behind it, produced by a script
  that's checked in.
- **Make the fast path legible.** Where speed matters, the reason it's
  fast should be explainable in a sentence (usually: "fixed shapes, so
  XLA compiles once").
- **Test the properties that catch real bugs**, not line coverage. A
  causal-mask leakage test is worth more than a hundred shape assertions.

Stack: Nx 0.13 + EXLA (CPU) for compute, Polaris for optimizers, Axon for
exactly one function (`categorical_cross_entropy`). Elixir 1.19 / OTP 27
on a 15-core Apple Silicon machine.

---

## Training runs

Every run we did, in order, and what each was for.

| # | Run | Config | Purpose | Outcome |
|---|---|---|---|---|
| 1 | Overfit-one-batch | 2L/2H/d32, vocab 64, 1 batch, 300 steps | Prove gradients flow end to end before trusting any real run | Loss < 0.5. Lives in the suite as `@tag :slow` |
| 2 | Throughput probe | `shakespeare_small`, 100 steps | Size the real run before committing hours to it | 21–25K tok/s → 5,000 steps ≈ 14 min. Discarded |
| 3 | **Shakespeare baseline** | `shakespeare_small`, 5,000 steps, 3.48M params | The main model: samples, attention analysis, generation benchmarks | Train 1.65 / **val 3.97**. Kept as `runs/20260903-081542` |
| 4 | **Variant A/B sweep** | 8 combos × 750 steps | Decide which architectural choices actually matter | RoPE and SwiGLU win; norm choice irrelevant. ~24 min total |
| 5 | Sizing probe | `tinystories_base`, 60 steps, 12.19M params | Check the big preset before committing | 5,400 tok/s → 20K steps would be **8.3 hours**. Rejected, discarded |
| 6 | **TinyStories controlled** | `tinystories_small`, 5,000 steps | Isolate *data volume*: identical architecture to run 3, 12× the tokens | See "Data volume" below |

Two of the six runs existed only to size a later run. That habit paid for
itself immediately at run 5, where the preset we'd written into the plan
would have burned 8 hours to teach us something a 90-second probe told us
for free.

### Why run 6 is shaped the way it is

The obvious next run after finding the overfitting in run 3 was the big
`tinystories_base` preset. That was the wrong experiment twice over.

At 12.19M parameters on 5.1M training tokens it would still be
data-starved (0.42 tokens/param), so it would have reproduced the same
overfitting result while changing four variables at once — depth, width,
context length, and data. Instead run 6 holds the architecture *identical*
to run 3 and changes only the corpus. The vocabulary necessarily changes
with it (1024 → 4096), which is a confound we can't remove, but it's one
confound instead of five.

---

## Findings: the model

### Data volume dominates everything else we measured

Run 3's headline number is not its validation loss but the **gap**: train
1.65 against val 3.97, with validation *rising* over the final 500 steps
(3.975 → 3.988 → 4.010). The model was memorizing Shakespeare, and the
best checkpoint (step 4,500) was already past the turn.

The cause is a data/parameter ratio of **0.118 tokens per parameter**
against a rule of thumb of ~20 — roughly 170× short.

There's an irony in how we got there. Choosing BPE *worsened* it:
compressing 1.1 MB into 456K tokens at 2.44 bytes/token threw away 2.4× of
the training signal that byte-level tokenization would have given. BPE is
the right trade when context length or compute per token is the binding
constraint. When **data** is the binding constraint it is actively
harmful, because it shrinks your token count for free.

Generalizable lesson: pick the tokenizer against the constraint you're
actually up against, and work out which that is before choosing.

### Which architecture choices matter (run 4)

Eight combinations, 750 steps each, identical everything else:

| pos | norm | mlp | params | val loss | train secs |
|---|---|---|---|---|---|
| rope | layer_norm | swiglu | 3,478,016 | 3.427 | 148 |
| rope | rms_norm | swiglu | 3,475,712 | **3.433** | **137** |
| rope | layer_norm | gelu | 3,412,480 | 3.529 | 220 |
| rope | rms_norm | gelu | 3,410,176 | 3.543 | 219 |
| learned | layer_norm | swiglu | 3,510,784 | 3.622 | 144 |
| learned | rms_norm | swiglu | 3,508,480 | 3.625 | 143 |
| learned | layer_norm | gelu | 3,445,248 | 3.726 | 208 |
| learned | rms_norm | gelu | 3,442,944 | 3.728 | 211 |

- **RoPE beats learned positions by ~0.20 nats.** The largest single
  effect, and it *removes* parameters rather than adding them.
- **SwiGLU beats GELU by ~0.10 nats and trains ~35% faster.** The speed
  was a surprise — it has *more* matrices (three vs two). The cause is
  that our GELU is the exact `erf` formulation, which is expensive on CPU;
  a `tanh` approximation would likely close the speed gap while keeping
  the quality difference.
- **Norm choice is a wash** (< 0.01 nats, within noise). RMSNorm is
  cheaper to implement and has fewer parameters, so it wins on
  simplicity, not quality.
- The ordering is perfectly consistent: every RoPE row beats every
  learned row, and within each position type every SwiGLU row beats every
  GELU row. Two independent effects, no interaction.

### Attention specializes with depth

From `mix wf.attention` on run 3 (4 layers × 4 heads, 20-token prompt):

- **Layers 0–1**: no crisp specialization. High entropy (0.60–0.81), mean
  attention distance 3.8–6.2 tokens, and the only genuine attention sinks
  in the model (layer 1 heads 0 and 1, at 1.6× and 1.9× the uniform
  baseline).
- **Layers 2–3**: **previous-token heads emerge**. Five of the eight heads
  in these layers are classified previous-token or mostly-previous-token.
  The cleanest is L3H0: 0.556 of its mass exactly one step back, mean
  distance 1.9, entropy 0.411 — visible as a hard diagonal band in the
  heatmap.
- Sentence-final punctuation behaves differently from everything else: the
  `.` row spreads attention across the entire preceding sentence rather
  than looking locally.

So the model builds a diffuse mixing stage first and sharp positional
circuitry on top of it. With only four layers the picture is compressed
compared to what's reported for larger models, but the direction is the
same.

### The attention-sink metric was measuring the wrong thing

First version of the sink statistic was "mean mass on position 0". A
*uniform* causal head scored 0.34 on it and got labelled a sink.

The bias is structural: position 0 is visible to all `t` query rows, while
position `t-1` is visible to exactly one. So a perfectly uniform head puts
`H(t)/t` of its mass on position 0 — 0.34 at t=8, 0.18 at t=20 — purely
from the causal mask, with no sink behavior at all. Any diffuse head looks
like a sink under the raw metric.

Fixed by reporting sink mass as a **multiple of that uniform baseline**.
The corrected table is much cleaner: layers 0, 2 and 3 sit at 0.6–1.3×
(i.e. nothing), and only layer 1 shows real sink behavior. Half the
"sinks" in the first table were an artifact of my own metric.

Generalizable lesson: before reading a statistic as evidence of a
behavior, work out what value it takes under the *null* behavior. Ratios
against a baseline beat raw masses whenever the sample geometry is
uneven.

---

## Findings: performance

### Generation — three strategies, two orders of magnitude

100 tokens, run 3's model, block size 128:

| Strategy | ms/token | vs. best |
|---|---|---|
| **KV cache** | **1.06** | 1× |
| Fixed-shape recompute | 7.21 | 6.8× slower |
| Naive growing-shape recompute | 99.58 | 94× slower |

Three separate effects stack here, and it's worth keeping them distinct:

1. **Naive → fixed-shape (14×)** is purely about compilation. Running the
   forward pass on a growing sequence gives XLA a new shape every token,
   so it recompiles every token. Padding the context to a fixed
   `{1, block}` buffer and tracking the length as a scalar means one
   compilation total. The causal mask makes the padding provably
   invisible, which is what licenses the trick — and there's a test
   asserting exactly that.
2. **Fixed-shape → KV cache (6.8×)** is about arithmetic. The fixed-shape
   path still recomputes keys and values for all 128 positions to produce
   one token, even though position `p`'s key and value cannot change once
   written. Caching them takes the per-token cost from O(context²) to
   O(context).
3. Both are dwarfed by how bad the naive version is, which is a useful
   reminder that with a compiler in the loop, *shape stability* is a
   first-order performance concern and not a detail.

Two subtleties the cache turned up:

- **Cache the keys *after* RoPE rotation.** Rotating on write is what
  makes cached scores exactly equal to the full forward pass. Rotating on
  read would need the query's position too and gets the relative offsets
  wrong.
- **A KV cache cannot slide its context window.** Cached keys were
  encoded at their original absolute positions, so shifting them left
  silently corrupts the positional information — for RoPE *and* for
  learned embeddings. The recomputing path can slide because it re-encodes
  from scratch. So generation auto-selects: cache while the context fits,
  recompute beyond it. This is a genuine architectural tension, not an
  implementation shortcut, and it's why production systems reach for
  relative-position schemes or explicit cache-eviction policies.

The strongest correctness evidence: both paths produce **byte-identical
text** for the same seed, across seeds, temperatures and top-k settings.
That falls out of keeping sampling in a separate compiled function so
prefilling the prompt doesn't consume the PRNG stream.

### Training throughput

~22K tok/s for the 3.48M-parameter model (batch 32 × block 128), giving
5,000 steps in ~14 minutes. The whole step — forward, `value_and_grad`,
global-norm clip, AdamW with its schedule — traces into one jitted
function compiled once and called 5,000 times.

The batch pipeline matters more than it looks. Random offsets broadcast
against an iota give every window's absolute indices, and a single
`Nx.take` gathers the whole `{batch, block+1}` window matrix out of the
corpus tensor. Nothing crosses to the host per batch and the shapes never
change.

### Tokenizer pre-tokenization — 4× from a hybrid scanner

Pre-tokenization (splitting text into words/numbers/punctuation/whitespace
before BPE proper) was a pure regex. Replacing it with a hybrid:

| | before | after | |
|---|---|---|---|
| Chunk 1.1 MB | 124 ms | **29 ms** | 4.3× |
| Chunk 22 MB | 2,679 ms | **710 ms** | 3.8× |
| `encode` 1.1 MB | 179 ms | **87 ms** | 2.1× |
| `encode` 22 MB | 2,396 ms | **1,162 ms** | 2.1× |
| `BPE.train` | 5,740 ms | 5,721 ms | unchanged |

Three ingredients:

1. **Slice at safe boundaries.** Cut every 64 KB at the first whitespace
   byte following a non-whitespace byte, where no chunk can straddle.
   Keeps intermediate lists small and short-lived.
2. **ASCII fast path.** Binary pattern matching over character classes is
   ~7× faster than the regex at the same job.
3. **Escape to the regex per-run, not per-slice.** See below — getting
   this wrong made things slower.

`BPE.train` is unchanged because learning the merges dominates it: the
frequency-table phase got 2× faster (244 → 117 ms) but it's only ~3% of
the 5.7s. Incremental pair counts in the merge loop are the next real win
there, if ever needed.

The regex stays as the **specification**. A differential test keeps its
own copy and asserts the fast paths agree byte-for-byte across tricky
ASCII cases, non-ASCII text, slice boundaries, the full real corpus, and a
property over arbitrary UTF-8. That test is what makes the optimization
safe to keep.

---

## Findings: how to measure things (three mistakes)

These cost the most time and are the most transferable.

### 1. I answered the right question with the wrong denominator

I measured pre-tokenization as **3% of `BPE.train`** and used that to argue
the regex didn't matter. But `train` is dominated by the merge search;
`encode` is dominated by chunking, where the same work is **68%**. Same
code, same machine, 20× different answer depending on which function you
put in the denominator.

A percentage is meaningless without naming its scope, and "it's only N% of
X" is only an argument if X is what the user is waiting on. The full
picture:

| Scope | Chunking share |
|---|---|
| `BPE.train` | ~3% |
| `BPE.encode` (Shakespeare) | 68% |
| `BPE.encode` (TinyStories) | ~76% |
| Whole tokenizer stage | 5–8% |
| Whole pipeline incl. training | <1% |

### 2. Benchmark contamination invented a 3× regression

I reported `BPE.train` regressing from 5.7s to 16.8s. It hadn't. The
benchmark script had allocated 22 MB corpora and multi-million-element
lists *before* calling `train`, leaving the process heap in a state that
made the measurement meaningless. Running `train` twice in a fresh process
gave 8.5s then 5.8s.

Every number in the tables above is now measured in an **isolated
process**, one operation per `mix run`. In a garbage-collected runtime, an
earlier measurement in the same process is part of the experimental setup
whether you intended it or not.

### 3. Coarse-grained fallback made the fast path slower

The first ASCII fast path escaped to the regex for a whole 64 KB slice on
encountering any byte ≥ 128. On TinyStories this was **slower than the
original** (2,679 → 3,105 ms).

The measurement explained it immediately: only **0.06% of TinyStories
bytes are non-ASCII**, but they're distributed evenly enough that **88% of
64 KB slices contain at least one**. So nearly every slice took the regex
path *and* paid for the aborted ASCII scan first. Making the escape
per-run instead of per-slice turned a 1.16× regression into a 3.8× win.

Lesson: for a fast-path/slow-path design, the thing to measure is not the
rate of the rare case but **how often the fast path is denied**. Those
differ by three orders of magnitude here.

A related incidental result, worth knowing on the BEAM: `Regex.scan` over
one 22 MB binary takes 2,679 ms, while scanning the same bytes
document-by-document takes 1,814 ms — **1.5× faster for identical work**,
purely from not materializing a 5.5-million-element result list.

---

## Findings: Nx and Elixir specifics

Things that cost real debugging time.

- **`Nx.select/3` needs its predicate broadcast to the full shape.** A
  `{t, t}` causal mask against `{b, h, t, t}` scores raises rather than
  broadcasting. `Nx.broadcast(mask, Nx.shape(scores))` first.
- **Tensor structs don't compare with `==`.** Two tensors holding the same
  value have different backend references, so `assert Nx.all_close(a, b)
  == Nx.tensor(1, type: :u8)` fails confusingly. Use
  `Nx.to_number(...) == 1`.
- **Module attributes can't hold tensors.** `@data Nx.iota({1000})` fails
  with "cannot escape #Reference" — attributes are escaped at compile
  time and a tensor holds a runtime reference. Use a function.
- **`Regex.scan` with `/u` raises on invalid UTF-8.** `<<72, 105, 128,
  33>>` is an `ArgumentError`, not a non-match. Any code promising to
  handle arbitrary binaries needs a non-regex path.
- **PCRE `\s` here is ASCII-only and includes `\v` (0x0B) and `\f`
  (0x0C).** Elixir's `/u` sets `PCRE_UTF8` but not `PCRE_UCP`, so `\s`
  doesn't grow to Unicode whitespace. My first hand-written character
  class omitted `\v` and `\f`; the differential test caught it.
- **Index versus length off-by-one.** My first hand-rolled scanner had
  `run_end` return an absolute index while the caller treated it as a
  length. It produced `" Citizen:"` where the spec gives `" Citizen"`,
  `":"` — wrong on the second chunk of the corpus, and it would have
  silently trained a different tokenizer. Differential testing found it in
  seconds; eyeballing would not have.
- **EXLA logs `Falling back to 1 / sqrt(x) for f32`** on CPU. Cosmetic,
  emitted by `Nx.rsqrt`.
- **`Nx.Defn.jit` over a closure capturing the config works well.** The
  config is read at trace time, so each variant combination compiles to
  its own specialized program with no runtime branching. This is what
  makes "every architectural choice is a flag" free at runtime.

---

## Findings: library API notes

- **We use almost nothing from Axon.** The plan was to drive training with
  `Axon.Loop`; the implementation hand-rolls the loop in ~60 lines
  (`Enum.reduce` over the batch stream around a jitted train step) and
  imports Axon solely for `Axon.Losses.categorical_cross_entropy`. The
  hand-rolled version is more legible for a learning project and we never
  missed the framework. Worth knowing before adding the dependency
  elsewhere.
- **Why explicit `defn` over the Axon graph API**: weight tying (reusing
  the token embedding as the output projection) has no first-class support
  in the graph API; attention, RoPE and SwiGLU are custom layers either
  way; and KV-cache decoding plus fused sampling are far easier with an
  explicit parameter map. The parameters being a visible nested map is
  also the whole pedagogical point.
- **`Polaris.Optimizers.adamw(learning_rate: fun)` accepts a 1-arity
  schedule function** of the step count. Not obvious from the docs;
  confirmed in the source.
- **Polaris has no schedule composition.** `Polaris.Schedules` has
  `cosine_decay` and a `:warmup` option on `linear_decay` only, so
  warmup-then-cosine is a ~10-line `defn` (`Warpweft.Schedule`). Cast the
  step to f32 first — it arrives as an integer tensor.
- **Gradient clipping composes left-to-right**:
  `clip_by_global_norm() |> Polaris.Updates.compose(adamw(...))` clips raw
  gradients before Adam scaling, which is the order you want.
- **Polaris `:decay` applies to every parameter**, including norm gains
  and embeddings. Selective weight decay (the usual practice of excluding
  norms and biases) isn't expressible without a custom updater. We
  accepted this; it's a real deviation from standard practice.
- **`Axon.rms_norm/2` exists** in Axon 0.8, but we hand-wrote both norms
  as four-line functions since seeing the arithmetic is the point.

---

## Open questions and next steps

- **Selective weight decay.** Excluding norms and embeddings needs a
  custom Polaris updater. Unknown whether it matters at this scale.
- **`tanh`-approximate GELU.** Would likely recover most of SwiGLU's
  35% speed advantage and isolate how much of run 4's MLP result was
  quality versus CPU cost of `erf`.
- **Incremental pair counts in BPE merge learning** — the remaining ~97%
  of `BPE.train`.
- **Cache eviction / sliding for KV decoding**, which needs a
  relative-position scheme to be correct.
- **Bits-per-byte evaluation.** Comparing a char-level model against a BPE
  model is invalid in nats/token because the vocabularies differ; the
  losses must be normalized to bits per byte. Worth building into the eval
  code *before* we need it, since it's an easy trap.
- **Scaling-law sweep**: train several sizes, plot loss against
  parameters and compute. The cleanest way to see the data/parameter
  tradeoff we backed into empirically.
