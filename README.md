# Warpweft

A decoder-only transformer built from scratch in Elixir, to learn how the
architecture actually works: Nx + EXLA for compute, a from-scratch
byte-level BPE tokenizer, a hand-rolled training loop, and swappable
architecture variants you can measure against each other.

Nothing is hidden behind a framework's model graph. The parameters are a
plain nested map, the forward pass is readable top-to-bottom Nx, and the
training step is four visible lines: forward, gradient, clip, update.

## Quick start

```sh
mix deps.get

mix wf.data --corpus shakespeare              # download corpus -> data/raw/
mix wf.tokenizer.train --corpus shakespeare --vocab 1024
                                              # train BPE + pre-tokenize -> data/tokenized/
mix wf.train --preset shakespeare_small       # ~14 min on a modern CPU -> runs/<timestamp>/
mix wf.generate --prompt "ROMEO:" -n 300      # sample from the latest run
```

TinyStories (richer English, 22 MB corpus, vocab 4096):

```sh
mix wf.data --corpus tinystories
mix wf.tokenizer.train --corpus tinystories --vocab 4096 --sample-mb 5
mix wf.train --preset tinystories_base
```

## What's in the box

| Piece          | Where                              | Notes                                                                         |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------- |
| Byte-level BPE | `lib/warpweft/tokenizer/bpe.ex`    | trained from scratch; any binary round-trips exactly; hybrid ASCII/regex pre-tokenizer |
| Data pipeline  | `lib/warpweft/data/`               | corpus -> u16 token file -> one device tensor; batches sampled fully on device |
| Model          | `lib/warpweft/model.ex` + `model/` | params are a visible nested map; forward pass is readable top-to-bottom Nx    |
| Attention      | `lib/warpweft/model/attention.ex`  | fused causal multi-head self-attention                                        |
| RoPE           | `lib/warpweft/model/rope.ex`       | rotary positions, half-split style                                            |
| Training       | `lib/warpweft/train.ex`            | hand-rolled loop: `value_and_grad` -> clip -> AdamW, one jitted step          |
| LR schedule    | `lib/warpweft/schedule.ex`         | linear warmup + cosine decay                                                  |
| Generation     | `lib/warpweft/generate.ex`         | compile-once fixed-shape sampling, Gumbel-max + top-k                         |
| Checkpoints    | `lib/warpweft/checkpoint.ex`       | self-contained `runs/<timestamp>/` dirs, resume with `--resume`               |

## Architecture variants

Every architectural choice is a flag, so you can measure one against the
other instead of taking it on faith. Defaults are the modern stack:

| Axis        | Modern (default)           | Classic                 |
| ----------- | -------------------------- | ----------------------- |
| Positions   | RoPE (rotary, zero params) | learned embedding table |
| Norm        | RMSNorm (pre-norm)         | LayerNorm (pre-norm)    |
| MLP         | SwiGLU                     | GELU 4x                 |
| Output head | tied to token embedding    | separate (`--no-tie`)   |

Run the classic configuration with
`--pos learned --norm layer --mlp gelu`. Variants are resolved at trace
time, so every combination compiles to its own specialized XLA program —
zero runtime branching.

### A/B results (shakespeare_small, equal steps)

Produced by `scripts/ab_variants.exs` on a 15-core Apple-Silicon CPU:

| pos     | norm       | mlp    | params    | val loss @750 | train secs |
| ------- | ---------- | ------ | --------- | ------------- | ---------- |
| rope    | layer_norm | swiglu | 3,478,016 | 3.427         | 148        |
| rope    | rms_norm   | swiglu | 3,475,712 | **3.433**     | **137**    |
| rope    | layer_norm | gelu   | 3,412,480 | 3.529         | 220        |
| rope    | rms_norm   | gelu   | 3,410,176 | 3.543         | 219        |
| learned | layer_norm | swiglu | 3,510,784 | 3.622         | 144        |
| learned | rms_norm   | swiglu | 3,508,480 | 3.625         | 143        |
| learned | layer_norm | gelu   | 3,445,248 | 3.726         | 208        |
| learned | rms_norm   | gelu   | 3,442,944 | 3.728         | 211        |

Takeaways at this scale: **RoPE beats learned positions by ~0.2 nats**,
**SwiGLU beats GELU by ~0.1 nats** (and trains ~35% faster here — exact
`erf`-based GELU is expensive on CPU), and the **norm choice is a wash**.
The modern defaults win on both axes that matter.

## Design notes: keeping XLA compiling once

Every hot path is shaped so the compiler sees fixed shapes and compiles a
single program, which is where most of the performance comes from.

1. **Batch sampling stays on device.** Random offsets broadcast against an
   iota give every window's absolute indices, and one `Nx.take` gathers
   the whole `{batch, block+1}` window matrix straight out of the corpus
   tensor. No host round-trip per batch, no per-batch stacking
   (`lib/warpweft/data/batches.ex`).
2. **One jitted training step.** Forward, `value_and_grad`, global-norm
   clip, and AdamW with its LR schedule all trace into a single function,
   compiled once and called `total_steps` times.
3. **One jitted generation step.** The context lives in a fixed-shape
   `{1, block}` right-padded buffer with a scalar length. Because the
   causal mask makes positions past the length unreachable, the padding is
   mathematically invisible — so sampling a token never changes the shape
   and never triggers a recompile. Growing the sequence instead would cost
   a fresh compilation at every length.
4. **Batched, correct sampling.** Temperature, top-k masking
   (`Nx.top_k`), and the sample itself happen inside the compiled step.
   Sampling uses the Gumbel-max trick — `argmax(logits/T + gumbel)` _is_ a
   draw from `softmax(logits/T)` — which is fully batched and needs no
   loop over the batch. Dropout draws a fresh PRNG key every step, so no
   two steps ever share a mask.

### Measured

On a 15-core Apple-Silicon CPU, `shakespeare_small` (3.5M params, RoPE +
RMSNorm + SwiGLU + tied):

- **Training**: ~22K tokens/s; the full 5,000-step run takes ~14 min
  (final val loss ≈ 3.97, i.e. ~1.63 nats/byte at 2.44 bytes/token)
- **Generation**: one 0.16s compilation, then **6 ms/token**. The
  recompile-per-length alternative costs 97 ms/token even at tiny context
  sizes and gets worse as the sequence grows — a **≥16× speedup**
  (`scripts/bench_generate.exs`)
- **Tokenizer**: 768 merges learned in ~6s; encodes Shakespeare at 2.44
  bytes/token. Pre-tokenization runs at ~32 MB/s (1.1 MB in 29 ms, the
  22 MB corpus in 710 ms), so `encode` over the whole 22 MB corpus takes
  1.2s. Learning the merges dominates training time, and is where to look
  next for a speedup

Sample after 14 minutes of CPU training (prompt `ROMEO:`):

```
ROMEO:
O, thy fiery day!

ROMEO:
Why, I can tell what:
O, then, you think, I say, is not so.

Servant:
I will, sir; this is come.

ROMEO:
Faith, be true:
Good Fortune's vouch of the care,
To executioner, and, to fight the sea,
In all the bent; and then, 'tis a garland.
```

## Tests

```sh
mix test                 # fast suite
mix test --include slow  # + overfit-one-batch gradient sanity, checkpoint round-trip
```

The suite covers the properties that actually catch transformer bugs:

- **causal-mask leakage**: mutate tokens after position _t_; logits at ≤ _t_
  must not move (run for both RoPE and learned positions)
- **padding safety**: logits at `len-1` identical under zero or garbage
  padding (what fixed-shape generation relies on)
- tokenizer round-trip property tests over arbitrary UTF-8 _and_ arbitrary
  binaries; deterministic training; save/load
- batch shift invariant (`y` is `x` one token left), index bounds
- sampler: temperature→0 equals argmax; samples never escape top-k support
- overfit-a-single-batch end-to-end (`@tag :slow`)

## Where to take it next

The architecture is task-agnostic — only the tokenizer and data change:

- **Algorithmic toys** (addition, sorting, reversal): tiny vocab, exact
  accuracy metrics instead of eyeballing prose
- **Music** in ABC notation
- **Code completion** on an Elixir corpus
- **KV-cache decoding**: an additive `decode_step` reusing the same params
  (the explicit-params design makes this a new module, not a rewrite)
- **EMLX backend** (Apple Metal) once its training support matures
