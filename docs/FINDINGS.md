# Warpweft: implementation log and findings

A running record of what we built, every training run and why we ran it,
what the measurements showed, and the mistakes worth remembering. Written
for the version of me that comes back to this in six months.

For the conceptual explanation of the architecture itself, see
[GUIDE.md](GUIDE.md). This document is the lab notebook, not the textbook.

---

## The goal

Learn how the transformer architecture actually works by building a
decoder-only language model from scratch in Elixir, with no framework
model graph, parameters visible as a plain nested map, and every
architectural choice a flag we can measure rather than a decision we take
on faith.

Secondary goals that shaped the design:

- **Measure, do not assert.** Every claim about architecture or
  performance in this repo should have a number behind it, produced by a
  script that is checked in.
- **Make the fast path legible.** Where speed matters, the reason it is
  fast should be explainable in a sentence (usually: "fixed shapes, so XLA
  compiles once").
- **Test the properties that catch real bugs**, not line coverage. A
  causal-mask leakage test is worth more than a hundred shape assertions.

Stack: Nx 0.13 + EXLA (CPU) for compute, Polaris for optimisers, Axon for
exactly one function (`categorical_cross_entropy`, called fully qualified;
there is no `import Axon` anywhere). Elixir 1.19 / OTP 27 on a 15-core
Apple Silicon machine.

---

## Training runs

Every run we did, in order, and what each was for.

| # | Run | Config | Purpose | Outcome |
|---|---|---|---|---|
| 1 | Overfit-one-batch | 2L/2H/d32, vocab 64, 1 batch, 300 steps | Prove gradients flow end to end before trusting any real run | Loss < 0.5. Lives in the suite as `@tag :slow` |
| 2 | Throughput probe | `shakespeare_small`, 100 steps | Size the real run before committing hours to it | 21-25K tok/s, so 5,000 steps ~15 min. Discarded |
| 3 | **Shakespeare baseline** | `shakespeare_small`, 5,000 steps, 3.48M params | The main model: samples, attention analysis, generation benchmarks | Best val **3.449**, risen to **4.000** by the end. `runs/20260903-081542` |
| 4 | **Variant A/B sweep** | 8 combos x 750 steps | Decide which architectural choices actually matter | RoPE and SwiGLU win; norm choice irrelevant. ~24 min total |
| 5 | Sizing probe | `tinystories_base`, 60 steps, 12.19M params | Check the big preset before committing | 5,400 tok/s, so 20K steps would be **8.3 hours**. Rejected, discarded |
| 6 | **TinyStories controlled** | `tinystories_small`, 5,000 steps, 4.26M params | Isolate *data volume*: identical architecture to run 3, twelve times the tokens | Best val **2.012**, gap **0.139** against Shakespeare's 0.754. `runs/20260904-173641` |

Two of the six runs existed only to size a later run. That habit paid for
itself immediately at run 5, where the preset we had written into the plan
would have burned eight hours to teach us something a 90-second probe told
us for free.

Run 6 was killed by an external signal at step 4,500 of 5,000 and resumed
from its checkpoint with `mix wf.train --resume runs/20260904-173641`,
which was the first real exercise of the resume path. It picked up at
exactly step 4,500 and finished cleanly. Note the directory argument:
`--resume` on its own parses to nothing and silently starts a *fresh* run.

### Why run 6 is shaped the way it is

The obvious next run after finding the overfitting in run 3 was the big
`tinystories_base` preset. That was the wrong experiment twice over.

At 12.19M parameters on 5.1M training tokens it would still be
data-starved (0.42 tokens per parameter), so it would have reproduced the
same overfitting result while changing four variables at once: depth,
width, context length, and data. Instead run 6 holds the architecture
*identical* to run 3 and changes only the corpus. The vocabulary
necessarily changes with it (1024 to 4096), which is a confound we cannot
remove, but it is one confound instead of five.

---

## Findings: the model

### Data volume dominates everything else we measured

Run 3 overfits. Its best validation loss arrives early in training, and by
step 5,000 validation has climbed to 4.000 while training loss keeps
falling. The cause is a data/parameter ratio of **0.118 tokens per
parameter** against a rule of thumb of ~20, roughly 170 times short.

There is an irony in how we got there. Choosing BPE *worsened* it:
compressing 1.1 MB into 456K tokens at 2.44 bytes per token threw away
2.4x of the training signal that byte-level tokenisation would have given.
BPE is the right trade when context length or compute per token is the
binding constraint. When **data** is the binding constraint it is actively
harmful, because it shrinks your token count for free.

Generalisable lesson: pick the tokenizer against the constraint you are
actually up against, and work out which that is before choosing.

### Run 6 confirmed it: the gap collapsed

Same architecture, same hyperparameters, same step count. Only the corpus
changed, and the vocabulary it forces. Both columns measured identically:
best checkpoint, inference mode, 40 batches on each split.

| | Shakespeare (run 3) | TinyStories (run 6) |
|---|---|---|
| Training tokens | 410,727 | 5,104,650 |
| Parameters | 3,475,712 | 4,262,144 |
| Tokens per parameter | 0.118 | **1.198** |
| Train loss | 2.695 | 1.873 |
| Validation loss | 3.449 | 2.012 |
| **Train/val gap** | **0.754** | **0.139** |
| Validation at end of run | risen to 4.000 | 2.012, still falling |
| Throughput | 22K tok/s | 17K tok/s |

The generalisation gap shrank more than fivefold. Validation decreased at
every one of run 6's twenty evaluations, so it ends **undertrained**
rather than overfit: the exact opposite regime, reached purely by feeding
it more data. Run 3 should have been stopped early; run 6 would have kept
improving past 5,000 steps.

Throughput dropped from 22K to 17K tok/s, from the four-times-larger
vocabulary making the final logit projection more expensive
(`batch x block x d x vocab`).

Sample quality tracks the loss. Run 3 produced Shakespeare-flavoured word
salad with correct-looking speaker labels. Run 6, prompt "Once upon a
time":

> Once upon a time, there was a big gray cat. The cat liked to sleep all
> day long. One day, the cat would sleep all day. It felt ashamed.
> The cat woke up and saw a little mouse. The mouse said, "Why are you
> sad, little mouse?" The mouse said, "I am sad because I need to clean."

Syntax, dialogue punctuation and register are essentially correct.
Coherence fails at the semantic level (the mouse asks the mouse why it is
sad), which is what a 4M-parameter model at 1.2 tokens per parameter
should look like.

### How good is it, in absolute terms?

Calibrated against baselines on the identical validation bytes (2.25 MB,
567K tokens), everything in bits per byte so it is comparable:

| | bits/byte |
|---|---|
| Uniform over the vocabulary | 3.029 |
| `gzip -9` | 2.386 |
| Unigram, fitted on train | 2.141 |
| `xz -9` | 1.729 |
| Bigram, fitted on train, interpolated | 1.347 |
| **Run 6** | **0.732** |

Comfortably better than general-purpose compressors and an n-gram model,
which is the classic demonstration that language modelling is compression.
Resist reading 0.732 as good in the absolute: TinyStories is deliberately
simple text, so this is not comparable to published figures on harder
corpora.

The consistent failure mode across samples is **entity tracking**: "Buzzy"
the bird becomes "Bobo", "Sara" becomes "Mia", an owl "was not frightened.
It was scared of the big bird" in consecutive sentences. That has a
mechanistic explanation in the attention analysis below: there are
previous-token heads but no induction heads, and induction heads are
exactly the circuit that copies a name forward.

### Comparing across tokenizers requires bits per byte

The two validation losses above, 3.449 and 2.012, are **not** comparable
as absolute numbers, and the temptation to read the second as "better" is
the trap. A tokenizer packing more text into each token earns a higher
per-token loss for identical predictive quality, so nats per token is
meaningless across different vocabularies.

Dividing that out (`Warpweft.Train.bits_per_byte/2`):

| Run | bytes/token | val nats/token | **bits/byte** |
|---|---|---|---|
| Shakespeare, vocab 1024 | 2.444 | 3.449 | **2.036** |
| TinyStories, vocab 4096 | 3.967 | 2.012 | **0.732** |

The gap is even wider on a fair footing, but this still is not a clean
statement about model quality, because the *corpora* differ in intrinsic
difficulty. TinyStories is deliberately simple English with a small
conceptual vocabulary; Shakespeare is archaic verse. Some of that
difference is the task being easier, not the model being better. Isolating
model quality would need both models trained on the *same* corpus with
different tokenizers.

Two confounds, one controlled: run 6 isolates data volume cleanly, because
the gap is a within-run measure where tokenizer differences cancel, but it
says nothing rigorous about absolute quality.

### Which architecture choices matter (run 4)

Eight combinations, 750 steps each, identical everything else:

| pos | norm | mlp | params | val loss | train secs |
|---|---|---|---|---|---|
| rope | layer_norm | swiglu | 3,478,016 | 3.427 | 148 |
| rope | rms_norm | swiglu | 3,475,712 | 3.433 | 137 |
| rope | layer_norm | gelu | 3,412,480 | 3.529 | 220 |
| rope | rms_norm | gelu | 3,410,176 | 3.543 | 219 |
| learned | layer_norm | swiglu | 3,510,784 | 3.622 | 144 |
| learned | rms_norm | swiglu | 3,508,480 | 3.625 | 143 |
| learned | layer_norm | gelu | 3,445,248 | 3.726 | 208 |
| learned | rms_norm | gelu | 3,442,944 | 3.728 | 211 |

- **RoPE beats learned positions by ~0.19 nats.** The largest single
  effect, and it *removes* parameters rather than adding them.
- **SwiGLU beats GELU by ~0.11 nats and trains ~37% faster.** The speed
  was a surprise, since it has *more* matrices (three against two). The
  cause is that our GELU is the exact `erf` formulation, which is
  expensive on CPU; a `tanh` approximation would likely close the speed
  gap while keeping the quality difference.
- **Norm choice is a wash** (under 0.01 nats, inside noise). RMSNorm is
  cheaper to implement and has fewer parameters, so it wins on simplicity,
  not quality.
- The ordering is perfectly consistent: every RoPE row beats every learned
  row, and within each position type every SwiGLU row beats every GELU
  row. Two independent effects, no interaction.

### Attention specialises with depth

From `mix wf.attention --run runs/20260903-081542` on the 20-token prompt
`"First Citizen:\nBefore we proceed any further, hear me speak."` (the
task's shorter default prompt gives a different table, so the prompt
matters when reproducing this):

- **Layers 0 and 1**: no crisp specialisation. High entropy (0.60 to
  0.81), mean attention distance 3.8 to 6.2 tokens, and the only genuine
  attention sinks in the model (layer 1 heads 0 and 1, at 1.6x and 1.9x
  the uniform baseline).
- **Layers 2 and 3**: **previous-token heads emerge**. Six of the eight
  heads in these layers are classified previous-token or
  mostly-previous-token. The cleanest is layer 3 head 0: 0.556 of its mass
  exactly one step back, mean distance 1.9, entropy 0.411, visible as a
  hard diagonal band in the heatmap.
- Sentence-final punctuation behaves differently from everything else: the
  `.` row spreads attention across the entire preceding sentence rather
  than looking locally.

So the model builds a diffuse mixing stage first and sharp positional
circuitry on top of it. With only four layers the picture is compressed
compared to what is reported for larger models, but the direction is the
same.

### The attention-sink metric was measuring the wrong thing

The first version of the sink statistic was "mean mass on position 0". A
*uniform* causal head scored 0.34 on it and got labelled a sink.

The bias is structural: position 0 is visible to all `t` query rows, while
position `t-1` is visible to exactly one. So a perfectly uniform head puts
`H(t)/t` of its mass on position 0 (0.34 at t=8, 0.18 at t=20) purely from
the causal mask, with no sink behaviour at all. Any diffuse head looks
like a sink under the raw metric.

Fixed by reporting sink mass as a **multiple of that uniform baseline**.
The corrected table is much cleaner: layers 0, 2 and 3 sit between 0.6x
and 1.5x, and only layer 1 shows real sink behaviour. Half the "sinks" in
the first table were an artefact of my own metric.

Generalisable lesson: before reading a statistic as evidence of a
behaviour, work out what value it takes under the *null* behaviour. Ratios
against a baseline beat raw masses whenever the sample geometry is uneven.

---

## Findings: performance

### Generation: three strategies, two orders of magnitude

100 tokens, run 3's model, block size 128:

| Strategy | ms/token | vs. best |
|---|---|---|
| **KV cache** | **1.06** | 1x |
| Fixed-shape recompute | 7.21 | 6.8x slower |
| Naive growing-shape recompute | 99.58 | 94x slower |

Three separate effects stack here, and it is worth keeping them distinct:

1. **Naive to fixed-shape (14x)** is purely about compilation. Running the
   forward pass on a growing sequence gives XLA a new shape every token,
   so it recompiles every token. Padding the context to a fixed
   `{1, block}` buffer and tracking the length as a scalar means one
   compilation total. The causal mask makes the padding provably
   invisible, which is what licenses the trick, and there is a test
   asserting exactly that.
2. **Fixed-shape to KV cache (6.8x)** is about arithmetic. The
   fixed-shape path still recomputes keys and values for all 128 positions
   to produce one token, even though position `p`'s key and value cannot
   change once written. Caching them takes the per-token cost from
   O(context²) to O(context).
3. Both are dwarfed by how bad the naive version is, which is a useful
   reminder that with a compiler in the loop, *shape stability* is a
   first-order performance concern and not a detail.

The naive row is measured over 15 tokens rather than 100, because it is
slow and gets slower with length, and it uses plain argmax rather than the
sampler. It is a speed reference, not a behavioural equivalent.

Two subtleties the cache turned up:

- **Cache the keys *after* RoPE rotation.** Rotating on write is what
  makes cached scores exactly equal to the full forward pass. Rotating on
  read would need the query's position too and gets the relative offsets
  wrong.
- **A KV cache cannot slide its context window.** Cached keys were encoded
  at their original absolute positions, so shifting them left silently
  corrupts the positional information, for RoPE *and* for learned
  embeddings. The recomputing path can slide because it re-encodes from
  scratch. So generation auto-selects: cache while the context fits,
  recompute beyond it. This is a genuine architectural tension, not an
  implementation shortcut, and it is why production systems reach for
  relative-position schemes or explicit cache-eviction policies.

The strongest correctness evidence: both paths produce **byte-identical
text** for the same seed, across seeds, temperatures and top-k settings.
That falls out of keeping sampling in a separate compiled function so
prefilling the prompt does not consume the PRNG stream.

### Training throughput

About 22K tok/s for the 3.48M-parameter model (batch 32 x block 128),
giving 5,000 steps in roughly 15 minutes. The whole step (forward,
`value_and_grad`, global-norm clip, AdamW with its schedule) traces into
one jitted function compiled once and called once per step.

The batch pipeline matters more than it looks. Random offsets broadcast
against an iota give every window's absolute indices, and a single
`Nx.take` gathers the whole `{batch, block+1}` window matrix out of the
corpus tensor. Nothing crosses to the host per batch and the shapes never
change.

### Tokenizer pre-tokenisation: 4x from a hybrid scanner

Pre-tokenisation (splitting text into words, numbers, punctuation and
whitespace before BPE proper) was a pure regex. Replacing it with a
hybrid:

| | before | after | |
|---|---|---|---|
| Chunk 1.1 MB | 124 ms | **29 ms** | 4.3x |
| Chunk 22 MB | 2,679 ms | **710 ms** | 3.8x |
| `encode` 1.1 MB | 179 ms | **87 ms** | 2.1x |
| `encode` 22 MB | 2,396 ms | **1,162 ms** | 2.1x |
| `BPE.train` | 5,740 ms | 5,721 ms | unchanged |

Three ingredients:

1. **Slice at safe boundaries.** Cut every 64 KB at a point where no chunk
   can straddle, keeping intermediate lists small and short-lived.
2. **ASCII fast path.** Binary pattern matching over character classes is
   about seven times faster than the regex at the same job.
3. **Escape to the regex per-run, not per-slice.** See below; getting this
   wrong made things slower.

`BPE.train` is unchanged because learning the merges dominates it: the
frequency-table phase got twice as fast (244 ms to 117 ms) but it is only
about 3% of the 5.7 seconds. Incremental pair counts in the merge loop are
the next real win there, if ever needed.

The regex stays as the **specification**. A differential test keeps its
own copy and asserts the fast paths agree byte-for-byte across tricky
ASCII cases, non-ASCII text, Unicode whitespace, slice boundaries, a
checked-in prose fixture, and a property over arbitrary UTF-8. That test
is what makes the optimisation safe to keep, and it is what eventually
caught the bug in the next section.

---

## Findings: how to measure things

These cost the most time and are the most transferable.

### 1. I answered the right question with the wrong denominator

I measured pre-tokenisation as **3% of `BPE.train`** and used that to argue
the regex did not matter. But `train` is dominated by the merge search;
`encode` is dominated by chunking, where the same work was **69%** before
the optimisation and **32%** after it. Same code, same machine, wildly
different answers depending on which function you put in the denominator.

A percentage is meaningless without naming its scope, and "it is only N%
of X" is only an argument if X is what the user is waiting on. Note also
that the "chunk 22 MB" and "encode 22 MB" rows in the table above are not
directly comparable as a ratio: `BPE.chunks/1` materialises all 5.5 million
chunks while `encode` streams them, so dividing one by the other gives a
nonsensical share above 100%.

### 2. Benchmark contamination invented a 3x regression

I reported `BPE.train` regressing from 5.7s to 16.8s. It had not. The
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
encountering any byte at or above 128. On TinyStories this was **slower
than the original** (2,679 ms to 3,105 ms).

The measurement explained it immediately: only **0.06% of TinyStories
bytes are non-ASCII**, but they are distributed evenly enough that **88%
of 64 KB slices contain at least one**. So nearly every slice took the
regex path *and* paid for the aborted ASCII scan first. Making the escape
per-run instead of per-slice turned a 1.16x regression into a 3.8x win.

Lesson: for a fast-path/slow-path design, the thing to measure is not the
rate of the rare case but **how often the fast path is denied**. Those
differ by three orders of magnitude here.

A related incidental result, worth knowing on the BEAM: `Regex.scan` over
one 22 MB binary takes 2,679 ms, while scanning the same bytes
document-by-document takes 1,814 ms, **1.5x faster for identical work**,
purely from not materialising a 5.5-million-element result list.

### 4. I published numbers read off a truncated log

The most embarrassing one, caught by a later review. I reported run 3's
validation loss as 3.975 and wrote a story about it "bottoming out at step
4,500". Both were wrong. The background job's output file had been
truncated to its last 30 lines, so the only three evaluations I could see
were the final three of twenty. The actual best checkpoint scores 3.449
and was written much earlier; 3.975 was simply the earliest number still
visible in the truncated file.

Worse, the "train/val gap" I derived from it (2.32) paired a
dropout-enabled training minibatch at step 5,000 with a validation loss
from step 4,500. Measured consistently, the gap is 0.754. The qualitative
conclusion survived, but three published numbers did not.

Lesson: a number you did not compute yourself, from a source you did not
check the completeness of, is a rumour. `best.ckpt` had the right answer
stored in it the whole time.

---

## Findings: Nx and Elixir specifics

Things that cost real debugging time.

- **`Nx.select/3` needs its predicate broadcast to the full shape.** A
  `{t, t}` causal mask against `{b, h, t, t}` scores raises rather than
  broadcasting. Call `Nx.broadcast(mask, Nx.shape(scores))` first.
- **`Nx.take` clamps out-of-range gather indices** rather than raising.
  This makes bounds assertions on gather indices vacuous, and lets a
  sampler that runs off the end of a corpus silently produce windows whose
  tails are one repeated token. Guard the range explicitly.
- **Tensor structs do not compare with `==`.** Two tensors holding the same
  value have different backend references, so
  `assert Nx.all_close(a, b) == Nx.tensor(1, type: :u8)` fails
  confusingly. Use `Nx.to_number(...) == 1`.
- **Module attributes cannot hold tensors.** `@data Nx.iota({1000})` fails
  with "cannot escape #Reference", because attributes are escaped at
  compile time and a tensor holds a runtime reference. Use a function.
- **`1..0` is a descending range.** `Enum.reduce_while(1..n, ...)` with
  `n = 0` iterates twice rather than not at all, which made
  `max_new_tokens: 0` generate two tokens. Write `1..n//1`.
- **`Regex.scan` with `/u` raises on invalid UTF-8.** `<<72, 105, 128, 33>>`
  is an `ArgumentError`, not a non-match. Any code promising to handle
  arbitrary binaries needs a non-regex path.
- **Elixir's `/u` modifier enables PCRE_UCP, so `\s` matches *Unicode*
  whitespace.** I believed and documented the opposite, and wrote an
  ASCII-only fast path whose "safe boundaries" a `\s+` chunk could
  straddle. The result was that a non-breaking space followed by an
  ordinary space chunked as two units instead of one, diverging from the
  specification the module claims to implement. Fixed by requiring both
  sides of a boundary to be ASCII. The `\v` and `\f` detail below is still
  true, it was the ASCII-only part that was wrong.
- **PCRE `\s` includes `\v` (0x0B) and `\f` (0x0C).** My first
  hand-written character class omitted both; the differential test caught
  it.
- **Index versus length off-by-one.** My first hand-rolled scanner had
  `run_end` return an absolute index while the caller treated it as a
  length. It produced `" Citizen:"` where the spec gives `" Citizen"`,
  `":"`. That was wrong on the second chunk of the corpus, and it would
  have silently trained a different tokenizer. Differential testing found
  it in seconds; eyeballing would not have.
- **`Nx.Random.randint` excludes its upper bound.** Our window sampler
  used `n - block - 1`, so the last corpus token was never predicted.
- **Streaming byte-level tokens needs a UTF-8 hold-back buffer.** A token
  can end mid-codepoint, so writing each token's bytes straight to the
  terminal prints mojibake. Incomplete trailing bytes must be held until
  the next token completes them. Bytes that can *never* form a character
  need substituting with U+FFFD rather than passing through; otherwise the
  stream can emit invalid UTF-8, which a test caught immediately with a
  randomly initialised model.
- **The BPE base alphabet is all 256 bytes regardless of corpus.** So
  every tokenizer's vocabulary contains raw high bytes, and a randomly
  initialised model emits malformed UTF-8 constantly. This defeated an
  attempt to test stream reassembly on an "ASCII-only" corpus: there is no
  such thing here. The working approach was a differential test mirroring
  the substitution policy.
- **`BPE.train/3`'s `vocab_size` is an upper bound, not a guarantee.**
  Merging halts early when the corpus runs out of adjacent pairs, which
  happens easily on small or repetitive text. Everything downstream must
  be named after the vocabulary *achieved*, not the one requested, or the
  tokenizer directory and the tokenized data end up under different names
  and training fails on a file no code path ever writes.
- **EXLA logs `Falling back to 1 / sqrt(x) for f32`** on CPU. Cosmetic,
  emitted by `Nx.rsqrt`.

---

## Findings: library API notes

- **We use almost nothing from Axon.** The plan was to drive training with
  `Axon.Loop`; the implementation hand-rolls the loop (about 80 lines: an
  `Enum.reduce` over the batch stream around a jitted train step) and
  touches Axon solely for one fully-qualified call to
  `Axon.Losses.categorical_cross_entropy`. The hand-rolled version is more
  legible for a learning project and we never missed the framework. Worth
  knowing before adding the dependency elsewhere.
- **Why explicit `defn` over the Axon graph API**: weight tying (reusing
  the token embedding as the output projection) has no first-class support
  in the graph API; attention, RoPE and SwiGLU are custom layers either
  way; and KV-cache decoding plus fused sampling are far easier with an
  explicit parameter map. The parameters being a visible nested map is
  also the whole pedagogical point.
- **`Polaris.Optimizers.adamw(learning_rate: fun)` accepts a 1-arity
  schedule function** of the step count. Not obvious from the docs;
  confirmed in the source.
- **Polaris silently ignores unknown optimiser options.** A renamed or
  mistyped key is dropped without error, which is why the optimiser tests
  observe the updates rather than trusting the construction.
- **Polaris has no schedule composition.** `Polaris.Schedules` has
  `cosine_decay` and a `:warmup` option on `linear_decay` only, so
  warmup-then-cosine is a short custom `defn` (`Warpweft.Schedule`). Cast
  the step to f32 first; it arrives as an integer tensor.
- **Gradient clipping composes left-to-right**:
  `clip_by_global_norm() |> Polaris.Updates.compose(adamw(...))` clips raw
  gradients before Adam scaling, which is the order you want.
- **Polaris `:decay` applies to every parameter**, including norm gains and
  embeddings. Selective weight decay (the usual practice of excluding
  norms and biases) is not expressible without a custom updater. We
  accepted this; it is a real deviation from standard practice.
- **`Axon.rms_norm/2` exists** in Axon 0.8, but we hand-wrote both norms as
  four-line functions since seeing the arithmetic is the point.

---

## Findings: what a review pass turned up

Late in the project we ran a deliberate review of the code, the tests and
the prose. It was worth more than any feature we could have added in the
same time, and the pattern generalises.

The most valuable technique was **mutation testing**: deliberately break
the implementation and check the suite goes red. Three things turned out
to be tested only in appearance.

- The batch sampler's "indices stay in range" test could not fail, because
  `Nx.take` clamps.
- Nothing connected the learning-rate schedule or weight decay to the
  optimiser. Replacing the schedule with a constant and zeroing the decay
  left all 59 tests green. The schedule was tested in isolation and the
  training test only checked that loss went down, which it does at any
  sane fixed rate.
- Temperature could be deleted entirely with no failures. The one test
  that mentioned it pinned `top_k: 1`, so temperature could not affect the
  outcome, and the cached-versus-recomputing test cancels it on both
  sides.

A fourth test claimed to exercise the KV-cache block limit but asked for
more tokens than fit, so it silently took the fallback path, and its
assertions (`is_binary/1`, `String.starts_with?`) held for any
implementation whatsoever.

Each of those now has a test that fails when the corresponding mutation is
applied, which was verified rather than assumed.

The review also found real bugs the tests had never covered: the
descending-range bug, the sampler off-by-one, the Unicode whitespace
divergence, a `FunctionClauseError` when no `runs/` directory exists,
resuming clobbering a better `best.ckpt` because the best-so-far was reset
to infinity, run directories colliding at one-second timestamp resolution,
and a benchmark script left broken by a refactor I never re-ran. That last
one is a special embarrassment: the repo states as a principle that every
performance claim should come from a checked-in script, and the script had
not run since the change.

Lesson worth keeping: a test suite passing tells you the tests pass. To
learn whether they *test* anything, break the code on purpose.

---

## Open questions and next steps

- **Test for induction heads.** The attention analysis found previous-token
  heads but no induction heads, and their absence explains the entity
  tracking failures. The standard probe is a repeated random sequence:
  check whether any head attends from each token to the position after its
  earlier occurrence. Cheap, and it would turn an inference into a
  measurement.
- **More data.** The full TinyStories training split is about 1.9 GB
  against the 22 MB validation file we trained on. Even 300 MB would give
  around 75M tokens, enough to scale to 10-20M parameters at a healthy
  ratio.
- **Train run 6 longer.** It ended still improving, so the 5,000-step
  budget (chosen for Shakespeare, which was overfitting by then) is now
  the binding constraint rather than the data.
- **Report bits per byte during training.** `bits_per_byte/2` exists, but
  the loop still logs nats per token, so every run's headline number is
  tokenizer-dependent.
- **Same corpus, two tokenizers.** The one comparison that would isolate
  tokenizer quality from corpus difficulty, and the missing control in the
  run 3 / run 6 pair.
- **A `tanh`-approximate GELU**, to separate how much of run 4's MLP result
  was quality and how much was the CPU cost of `erf`.
- **Selective weight decay**, excluding norms and embeddings, which needs a
  custom Polaris updater.
- **Incremental pair counts in BPE merge learning**, the remaining ~97% of
  `BPE.train`.
- **Cache eviction for KV decoding**, which needs a relative-position
  scheme to be correct.
- **A scaling-law sweep**: train several sizes, plot loss against
  parameters and compute. The cleanest way to see the data/parameter
  tradeoff we backed into empirically, and it wants the bigger corpus
  first.
