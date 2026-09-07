# Warpweft

A decoder-only transformer built from scratch in Elixir, to learn how the
architecture actually works: Nx + EXLA for compute, a from-scratch
byte-level BPE tokenizer, a hand-rolled training loop, and swappable
architecture variants you can measure against each other.

Nothing is hidden behind a framework's model graph. The parameters are a
plain nested map, the forward pass is readable top-to-bottom Nx, and the
training step is four visible lines: forward, gradient, clip, update.

## Documentation

| Document | What it is |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | What the model is: components, tensor shapes, data flow, and the variant matrix |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | What the words mean: a reference for every concept used, grouped by area |
| [docs/TECHNIQUES.md](docs/TECHNIQUES.md) | How it was built: parsing, array programming, compilation, testing and measurement practices |
| [docs/FINDINGS.md](docs/FINDINGS.md) | The lab notebook: every training run, what it measured, and the mistakes worth remembering |

Most of the techniques document has nothing to do with machine learning
and transfers directly to ordinary software.

## Getting started

Requires Elixir 1.19 or later and OTP 26 or later. Everything below runs
on CPU, no GPU needed. The first `mix deps.get` pulls EXLA, which is a
large download.

```sh
mix deps.get

mix wf.data --corpus shakespeare                          # download the corpus
mix wf.tokenizer.train --corpus shakespeare --vocab 1024  # ~6 seconds
mix wf.train --preset shakespeare_small                   # ~15 minutes
```

That gives you a trained 3.5M-parameter model in `runs/<timestamp>/`.
Then:

```sh
mix wf.repl
```

It loads the newest run, warms up the compiled sampler, and gives you a
prompt. Type anything and the continuation streams back token by token:

```
warpweft> Once upon a time

Once upon a time, there was a little girl named Amy. Amy loved to play
with her toys and share them with her friends. One day, Amy found a
small toy box in her room. The toy was a magic wand. It could talk!

[120 tokens, 96 ms, seed 952621]
```

Commands inside the prompt: `/temp 1.2`, `/n 200`, `/top-k 0`, `/seed 42`
(or `/seed random`), `/settings`, `/help`, `/quit`. On the KV-cache path
temperature is a runtime argument, so changing it is free; changing
`top-k` recompiles, because `Nx.top_k` needs `k` to shape its output when
the program is traced.

One-shot, without the REPL:

```sh
mix wf.generate --prompt "Once upon a time" -n 100
```

Asking for more than `block_size` minus the prompt drops to the slower
recomputing path, which is correct but about seven times slower per token
and does recompile when temperature changes.

## Looking inside

See one prompt traverse every stage of the model, with real shapes and
real probabilities:

```sh
mix wf.explain --prompt "The cat sat on the"
```

Inspect what the trained model's attention heads learned:

```sh
mix wf.attention --prompt "First Citizen:"            # per-head statistics
mix wf.attention --prompt "First Citizen:" --heatmaps # + shaded grids
```

Training writes checkpoints as it goes, so an interrupted run can be
picked up with `mix wf.train --resume runs/<timestamp>` (the directory
argument is required). All three tools take `--run <dir>` to select a
specific run rather than the newest.

TinyStories (richer English, 22 MB corpus, vocab 4096):

```sh
mix wf.data --corpus tinystories
mix wf.tokenizer.train --corpus tinystories --vocab 4096 --sample-mb 5
mix wf.train --preset tinystories_small   # same architecture as shakespeare_small
mix wf.train --preset tinystories_base    # 12.2M params, ~8h on CPU
```

## What's in the box

| Piece          | Where                              | Notes                                                                                  |
| -------------- | ---------------------------------- | -------------------------------------------------------------------------------------- |
| Byte-level BPE | `lib/warpweft/tokenizer/bpe.ex`    | trained from scratch; any binary round-trips exactly; hybrid ASCII/regex pre-tokenizer |
| Data pipeline  | `lib/warpweft/data/`               | corpus -> u16 token file -> one device tensor; batches sampled fully on device         |
| Model          | `lib/warpweft/model.ex` + `model/` | params are a visible nested map; forward pass is readable top-to-bottom Nx             |
| Attention      | `lib/warpweft/model/attention.ex`  | fused causal multi-head self-attention                                                 |
| RoPE           | `lib/warpweft/model/rope.ex`       | rotary positions, half-split style                                                     |
| Training       | `lib/warpweft/train.ex`            | hand-rolled loop: `value_and_grad` -> clip -> AdamW, one jitted step                   |
| LR schedule    | `lib/warpweft/schedule.ex`         | linear warmup + cosine decay                                                           |
| Generation     | `lib/warpweft/generate.ex`         | KV-cache decoding, Gumbel-max + top-k, UTF-8-safe token streaming                      |
| KV cache       | `lib/warpweft/model/decode.ex`     | single-position forward pass; O(context) per token instead of O(context²)               |
| Introspection  | `lib/warpweft/introspect.ex`       | per-head attention statistics and terminal heatmaps                                    |
| REPL           | `lib/mix/tasks/wf.repl.ex`         | interactive prompt, model loaded and compiled once                                     |
| Checkpoints    | `lib/warpweft/checkpoint.ex`       | self-contained `runs/<timestamp>/` dirs, resume with `--resume`                        |

## Architecture variants

Every architectural choice is a flag, so you can measure one against the
other instead of taking it on faith. Defaults are the modern stack:

| Axis        | Modern (default)           | Classic                 |
| ----------- | -------------------------- | ----------------------- |
| Positions   | RoPE (rotary, zero params) | learned embedding table |
| Norm        | RMSNorm (pre-norm)         | LayerNorm (pre-norm)    |
| MLP         | SwiGLU                     | GELU 4x                 |
| Output head | tied to token embedding    | separate                |

Run the classic configuration with
`--pos learned --norm layer --mlp gelu --no-tie`. Variants are resolved at
trace time, so every combination compiles to its own specialised XLA
program, with zero runtime branching.

### A/B results (shakespeare_small, equal steps)

Produced by `scripts/ab_variants.exs` on a 15-core Apple Silicon CPU.
Lowest validation loss first; the default configuration is marked.

| pos     | norm       | mlp    | params    | val loss @750 | train secs |          |
| ------- | ---------- | ------ | --------- | ------------- | ---------- | -------- |
| rope    | layer_norm | swiglu | 3,478,016 | 3.427         | 148        |          |
| rope    | rms_norm   | swiglu | 3,475,712 | 3.433         | 137        | default  |
| rope    | layer_norm | gelu   | 3,412,480 | 3.529         | 220        |          |
| rope    | rms_norm   | gelu   | 3,410,176 | 3.543         | 219        |          |
| learned | layer_norm | swiglu | 3,510,784 | 3.622         | 144        |          |
| learned | rms_norm   | swiglu | 3,508,480 | 3.625         | 143        |          |
| learned | layer_norm | gelu   | 3,445,248 | 3.726         | 208        |          |
| learned | rms_norm   | gelu   | 3,442,944 | 3.728         | 211        |          |

Takeaways at this scale: **RoPE beats learned positions by ~0.2 nats**,
**SwiGLU beats GELU by ~0.1 nats** (and trains ~35% faster here, because
the exact `erf`-based GELU is expensive on CPU), and the **norm choice is
a wash**, well inside noise. RMSNorm is the default on simplicity, not
quality.

### Data volume beats every architecture choice we measured

Same architecture, same hyperparameters, same 5,000 steps; only the corpus
differs. Both rows measured the same way: best checkpoint, inference mode,
40 batches on each split.

|                     | Shakespeare (410K tokens) | TinyStories (5.1M tokens) |
| ------------------- | ------------------------- | ------------------------- |
| Tokens per parameter | 0.118                    | 1.198                     |
| Train loss          | 2.695                     | 1.873                     |
| Validation loss     | 3.449                     | 2.012                     |
| **Train/val gap**   | **0.754**                 | **0.139**                 |
| Validation by end of run | risen to 4.000 (overfitting) | 2.012, still falling |

The Shakespeare run memorises: its best validation loss arrives early and
by step 5,000 validation has climbed to 4.000. Feeding the identical model
twelve times more data cuts the generalisation gap more than fivefold, and
validation was still improving when the budget ran out. For scale, the
~0.2 nats that RoPE bought was the largest architectural effect we
measured; data was worth an order of magnitude more.

The two validation losses are only comparable via bits per byte (2.036
against 0.732) because the vocabularies differ. See
[docs/FINDINGS.md](docs/FINDINGS.md) for why that matters and what it
still does not prove.

TinyStories sample after ~20 minutes of CPU training:

```
Once upon a time, there was a big gray cat. The cat liked to sleep all
day long. One day, the cat would sleep all day. It felt ashamed.
The cat woke up and saw a little mouse. The mouse said, "Why are you
sad, little mouse?" The mouse said, "I am sad because I need to clean."
```

## Design notes: keeping XLA compiling once

Every hot path is shaped so the compiler sees fixed shapes and compiles a
single program, which is where most of the performance comes from.

1. **Batch sampling stays on device.** Random offsets broadcast against an
   iota give every window's absolute indices, and one `Nx.take` gathers
   the whole `{batch, block+1}` window matrix straight out of the corpus
   tensor. No host round-trip per batch, no per-batch stacking. See
   `lib/warpweft/data/batches.ex`.
2. **One jitted training step.** Forward, `value_and_grad`, global-norm
   clip, and AdamW with its learning-rate schedule all trace into a single
   function, compiled once and called once per step.
3. **One jitted generation step.** Two layers of this. The KV cache means
   each token computes one position's keys and values and reads the rest
   from a buffer, rather than recomputing the whole context. Where the
   cache cannot be used (it cannot slide its window, because cached keys
   were encoded at their original absolute positions), the fallback keeps
   the context in a fixed-shape right-padded buffer with a scalar length.
   The causal mask makes the padding provably invisible, so sampling never
   changes a shape and never triggers a recompile.
4. **Batched, correct sampling.** Temperature, top-k masking via
   `Nx.top_k`, and the sample itself all happen inside the compiled step.
   Sampling uses the Gumbel-max trick, where `argmax(logits/T + gumbel)`
   _is_ a draw from `softmax(logits/T)`. It is fully batched and needs no
   loop over the batch. Dropout draws a fresh PRNG key every step, so no
   two steps ever share a mask.

### Measured

On a 15-core Apple Silicon CPU.

**Training**, `shakespeare_small` (3.48M params, RoPE + RMSNorm + SwiGLU +
tied): ~22K tokens/s, so the 5,000-step run takes about 15 minutes.
`tinystories_small` (4.26M params, vocab 4096) runs at ~17K tokens/s, the
larger vocabulary making the final logit projection more expensive.

**Generation**, 100 tokens at block size 128 on the Shakespeare model
(`scripts/bench_generate.exs`, which benchmarks whichever run is newest,
so rerun it against a specific `--run` to reproduce these exactly):

| Strategy                      | ms/token |
| ----------------------------- | -------- |
| KV cache                      | **1.06** |
| Fixed-shape recompute         | 7.21     |
| Naive growing-shape recompute | 99.58    |

The 14x from naive to fixed-shape is compilation: one XLA program instead
of one per sequence length. The further 6.8x from the KV cache is
arithmetic: it stops recomputing keys and values that cannot change. The
naive row is measured over 15 tokens rather than 100, because it is slow
and gets slower with length. The KV-cache and fixed-shape paths produce
byte-identical output for a given seed; the naive row uses plain argmax
and is a speed reference only.

**Tokenizer**: 768 merges learned in about 6 seconds; Shakespeare encodes
at 2.44 bytes per token. Pre-tokenisation runs at roughly 32 MB/s (1.1 MB
in 29 ms, the 22 MB corpus in 710 ms), so encoding the whole 22 MB corpus
takes 1.2 seconds. Learning the merges dominates tokenizer training and is
where to look next for a speedup.

## Tests

```sh
mix test                 # fast suite
mix test --include slow  # + overfit-one-batch gradient sanity
```

The suite covers the properties that actually catch transformer bugs:

- **causal-mask leakage**: mutate tokens after position _t_; logits at or
  before _t_ must not move, checked for both RoPE and learned positions
- **padding safety**: logits at `len-1` identical under zero or garbage
  padding, which is what fixed-shape generation relies on
- **KV cache equivalence**: cached decoding reproduces the full forward
  pass for all eight variant combinations, and both generation paths emit
  identical text across seeds, temperatures and top-k settings
- **RoPE relative-position invariance**: the score between a rotated query
  and key depends only on the distance between their positions
- tokenizer round-trip properties over arbitrary UTF-8 _and_ arbitrary
  binaries, plus a differential test pinning the fast chunker to its regex
  specification, including Unicode whitespace
- **optimiser wiring**: the learning-rate schedule, weight decay and
  gradient clipping are observed in the updates the optimiser produces,
  not just tested in isolation
- batch sampling invariants: windows are strictly consecutive, the final
  token is reachable, a too-short corpus raises
- sampler: temperature genuinely controls concentration; samples never
  escape the top-k support
- config round-trip, checkpoint and resume round-trip, and an assertion
  that `Config.atom_fields/0` matches the struct

Several of these exist because the originals were vacuous. Where a test
guards something subtle, the surrounding comment says which mutation it is
there to catch.

## Where to take it next

The architecture is task-agnostic: only the tokenizer and data change.

- **Algorithmic toys** (addition, sorting, reversal): tiny vocab, exact
  accuracy metrics instead of eyeballing prose
- **Music** in ABC notation
- **Code completion** on an Elixir corpus
- **EMLX backend** (Apple Metal) once its training support matures
- Further open questions are collected at the end of
  [docs/FINDINGS.md](docs/FINDINGS.md)

## Corpora

`mix wf.data` downloads from third parties rather than vendoring anything.
Neither corpus is redistributed here, and neither is covered by this
repository's licence.

**tinyshakespeare** (1.1 MB) comes from
[karpathy/char-rnn](https://github.com/karpathy/char-rnn), which is MIT
licensed. The underlying text is public-domain Shakespeare.

**TinyStories** comes from the
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset, released under
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) and introduced in
[*TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?*](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023). We use
the validation split as a CPU-scale training corpus and carve our own
validation split from it, which is deliberate and documented in
[docs/FINDINGS.md](docs/FINDINGS.md).

The rotary embedding implementation follows
[*RoFormer*](https://arxiv.org/abs/2104.09864) (Su et al., 2021), and the
BPE training procedure follows
[Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909).
