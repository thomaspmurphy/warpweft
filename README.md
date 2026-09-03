# Warpweft

A decoder-only transformer ("GPT from scratch") in Elixir, built to *learn
the architecture*: Nx + EXLA for compute, a from-scratch byte-level BPE
tokenizer, a hand-rolled training loop, and modern architecture variants
you can A/B against the nanoGPT baseline.

Inspired by Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY),
engineered rather than notebooked.

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

| Piece | Where | Notes |
|---|---|---|
| Byte-level BPE | `lib/warpweft/tokenizer/bpe.ex` | trained from scratch; any binary round-trips exactly |
| Data pipeline | `lib/warpweft/data/` | corpus -> u16 token file -> one device tensor; batches sampled fully on device |
| Model | `lib/warpweft/model.ex` + `model/` | params are a visible nested map; forward pass is readable top-to-bottom Nx |
| Attention | `lib/warpweft/model/attention.ex` | fused causal multi-head self-attention |
| RoPE | `lib/warpweft/model/rope.ex` | rotary positions, GPT-NeoX half-split style |
| Training | `lib/warpweft/train.ex` | hand-rolled loop: `value_and_grad` -> clip -> AdamW, one jitted step |
| LR schedule | `lib/warpweft/schedule.ex` | linear warmup + cosine decay |
| Generation | `lib/warpweft/generate.ex` | compile-once fixed-shape sampling, Gumbel-max + top-k |
| Checkpoints | `lib/warpweft/checkpoint.ex` | self-contained `runs/<timestamp>/` dirs, resume with `--resume` |

## Architecture variants

Defaults are the modern stack; each piece is a flag so you can A/B against
the nanoGPT baseline (`--pos learned --norm layer --mlp gelu`):

| Axis | Modern (default) | Baseline |
|---|---|---|
| Positions | RoPE (rotary, zero params) | learned embedding table |
| Norm | RMSNorm (pre-norm) | LayerNorm (pre-norm) |
| MLP | SwiGLU | GELU 4x |
| Output head | tied to token embedding | separate (`--no-tie`) |

Variants are resolved at trace time, so every combination compiles to its
own specialized XLA program — zero runtime branching.

### A/B results (shakespeare_small, equal steps)

_See `scripts/ab_variants.exs`; table filled from a run on an M-series CPU._

| pos | norm | mlp | params | val loss @750 | train secs |
|---|---|---|---|---|---|
| rope | layer_norm | swiglu | 3,478,016 | 3.427 | 148 |
| rope | rms_norm | swiglu | 3,475,712 | **3.433** | **137** |
| rope | layer_norm | gelu | 3,412,480 | 3.529 | 220 |
| rope | rms_norm | gelu | 3,410,176 | 3.543 | 219 |
| learned | layer_norm | swiglu | 3,510,784 | 3.622 | 144 |
| learned | rms_norm | swiglu | 3,508,480 | 3.625 | 143 |
| learned | layer_norm | gelu | 3,445,248 | 3.726 | 208 |
| learned | rms_norm | gelu | 3,442,944 | 3.728 | 211 |

Takeaways at this scale: **RoPE beats learned positions by ~0.2 nats**,
**SwiGLU beats GELU by ~0.1 nats** (and trains ~35% faster here — exact
`erf`-based GELU is expensive on CPU), and the **norm choice is a wash**.
The default modern stack is the right pick on both axes that matter.

## Performance: what this does differently

This project started from a Livebook port of Karpathy's video that worked
but was slow. The four fixes, all measurable:

1. **Batch sampling on device.** The livebook pulled random indices to the
   host (`Nx.to_list`) and rebuilt each batch with `Enum.map` + `Nx.slice` +
   `Nx.stack`. Here one `Nx.take` gathers the whole `{batch, block+1}`
   window matrix from the corpus tensor — fixed shapes, compiled once,
   no host round-trips (`lib/warpweft/data/batches.ex`).
2. **Compile-once training step.** The full step — forward,
   `value_and_grad`, global-norm clip, AdamW with its LR schedule — is one
   jitted function called `total_steps` times.
3. **Compile-once generation.** The livebook called `Axon.predict` on a
   growing sequence: a fresh XLA compilation per sequence length plus
   graph re-dispatch per token. Here the context lives in a fixed-shape
   `{1, block}` right-padded buffer (the causal mask makes padding
   invisible), so there is exactly one compilation, then every token is a
   single fast call (`scripts/bench_generate.exs` measures the difference).
4. **Correct, batched sampling.** Multinomial sampling via a `while` loop
   over the batch is replaced by the Gumbel-max trick —
   `argmax(logits/T + gumbel)` *is* a sample from `softmax(logits/T)` —
   plus top-k masking with `Nx.top_k`. And dropout gets a fresh PRNG key
   every step (the livebook reused a constant key, i.e. the same dropout
   mask forever).

### Measured

On a 15-core Apple-Silicon CPU, `shakespeare_small` (3.5M params, RoPE +
RMSNorm + SwiGLU + tied):

- **Training**: ~22K tokens/s; the full 5,000-step run takes ~14 min
  (final val loss ≈ 3.97, i.e. ~1.63 nats/byte at 2.44 bytes/token)
- **Generation**: one 0.16s compilation, then **6 ms/token**; the naive
  recompile-per-length approach costs 97 ms/token at tiny context sizes
  and grows with length — **≥16× speedup**

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

- **causal-mask leakage**: mutate tokens after position *t*; logits at ≤ *t*
  must not move (run for both RoPE and learned positions)
- **padding safety**: logits at `len-1` identical under zero or garbage
  padding (what fixed-shape generation relies on)
- tokenizer round-trip property tests over arbitrary UTF-8 *and* arbitrary
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
  (the explicit-defn design makes this a module, not a rewrite)
- **EMLX backend** (Apple Metal) once its training support matures
