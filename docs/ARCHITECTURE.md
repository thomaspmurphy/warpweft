# The transformer architecture

A structural reference for the decoder-only transformer implemented in
this repository: what the components are, what shape the data is at each
point, what each part contributes, and which choices are variable.

For the terminology, see [CONCEPTS.md](CONCEPTS.md). For the engineering
practices used to build it, see [TECHNIQUES.md](TECHNIQUES.md).

Throughout, the running example is the trained TinyStories model, so the
numbers are real rather than illustrative:

```
vocab_size  4096      block_size  128     n_layer  4
d_model     256       n_head      4       head_dim 64
mlp         swiglu, hidden 704            positions  RoPE
parameters  4,262,144                     embeddings tied
```

---

## 1. What the model computes

One function, applied to a sequence of token ids:

```
forward : {batch, seq} integers  ->  {batch, seq, vocab} scores
```

Position `i` of the output holds the model's scores for what token should
come *after* position `i` of the input. Every position produces a
prediction in the same pass, which is central to how training works and is
covered in section 6.

The whole model is in `lib/warpweft/model.ex`, and it is short enough to
read in one sitting. The parameters are a plain nested map, not a
framework object.

---

## 2. Data flow, with shapes

```
  tokens                                    {B, T}         integers

  |  embedding lookup
  v
  x                                         {B, T, 256}    the residual stream
  |
  |  +--------------------------------------------------+
  |  |  BLOCK, repeated 4 times                          |
  |  |                                                   |
  |  |    a = rms_norm(x)                {B, T, 256}     |
  |  |    a = attention(a)               {B, T, 256}     |
  |  |    x = x + a                      {B, T, 256}     |
  |  |                                                   |
  |  |    m = rms_norm(x)                {B, T, 256}     |
  |  |    m = swiglu(m)                  {B, T, 256}     |
  |  |    x = x + m                      {B, T, 256}     |
  |  +--------------------------------------------------+
  |
  v
  x = rms_norm(x)                           {B, T, 256}    final norm
  |
  |  project against the embedding table, transposed
  v
  logits                                    {B, T, 4096}
```

Notice that the residual stream stays `{B, T, 256}` from the embedding to
the final norm. Every block reads from it and adds back to it. Nothing
along that path changes shape, which is both a design property and the
reason the model is easy to reason about.

---

## 3. Embedding

A lookup table of one learned vector per vocabulary entry.

```
wte : {4096, 256}        1,048,576 parameters, 24.6% of the model
```

`Nx.take(wte, tokens)` turns `{B, T}` integers into `{B, T, 256}` floats.
There is no arithmetic here, just indexing, but the table is learned, so
tokens appearing in similar contexts drift towards similar vectors.

At this scale the embedding is a quarter of all parameters, entirely
because the vocabulary is large relative to the model. That ratio shifts
dramatically with size: in a large model the embedding is a rounding
error.

---

## 4. Position

Attention computes a weighted average, and averages are order-independent,
so without positional information `"dog bites man"` and `"man bites dog"`
would give identical outputs. Position must be added deliberately.

Two implementations, selectable with `pos:`.

### Learned (`pos: :learned`)

A second lookup table, one vector per position, added to the token
embedding:

```
wpe : {block_size, 256}      32,768 parameters at block 128
x = wte[tokens] + wpe[0..T-1]
```

Simple and effective. Costs parameters, and the model learns nothing about
positions beyond those it trained on.

### Rotary (`pos: :rope`, the default)

Nothing is added to the embedding. Instead, inside attention, each query
and key vector is **rotated** by an angle proportional to its position.
The head dimension is treated as `head_dim / 2` two-dimensional pairs, and
each pair is rotated:

```
rotated = [x1 * cos - x2 * sin,  x2 * cos + x1 * sin]
```

where the angle for position `p` and pair `i` is `p * theta^(-2i/head_dim)`
with `theta = 10000`. Low-index pairs rotate fast and encode fine
positional detail; high-index pairs rotate slowly and encode coarse
position.

The property that makes this worth the trouble: the dot product of a query
rotated by `i` against a key rotated by `j` depends only on `i - j`. The
attention scores see **relative** distance and never absolute position.
Zero parameters, and it measured about 0.19 nats better than learned
positions in our sweep, the largest architectural effect we found.

Implementation: `lib/warpweft/model/rope.ex`. Requires an even `head_dim`,
which `Config.validate!/1` enforces, because an odd one would silently
discard a channel.

---

## 5. Attention

`lib/warpweft/model/attention.ex`. One block's attention holds:

```
qkv.kernel  : {256, 768}     196,608     one matrix producing q, k and v
proj.kernel : {256, 256}      65,536     output projection
                             -------
                             262,144     6.2% of the model per block
```

### The computation

```
1.  qkv = x · W_qkv                          {B, T, 768}
2.  split into q, k, v                       {B, T, 256} each
3.  reshape to heads                         {B, 4, T, 64} each
4.  optionally rotate q and k (RoPE)
5.  scores = q · kᵀ / sqrt(64)               {B, 4, T, T}
6.  apply the causal mask
7.  weights = softmax(scores)                {B, 4, T, T}
8.  out = weights · v                        {B, 4, T, 64}
9.  merge heads back                         {B, T, 256}
10. out = out · W_proj                       {B, T, 256}
```

Step 5 is the heart of it. Every position's query is compared against
every position's key by dot product, producing a `T x T` grid of
relevance scores per head. Step 7 turns each row into a probability
distribution, and step 8 takes the corresponding weighted average of the
values.

### Why the scaling factor

Dot products of `d`-dimensional vectors grow with `d`. Without dividing by
`sqrt(head_dim)`, larger head dimensions push softmax into a regime where
one weight is nearly 1 and the rest nearly 0, which flattens the gradients
and stalls learning. The scaling keeps score magnitudes stable across
dimensions.

### The causal mask

```
mask[i][j] = i >= j
scores = select(mask, scores, -1.0e9)
```

Positions after `i` get a score of negative infinity, so softmax assigns
them zero weight. Position `i` can attend to `0..i` and no further.

This is what makes the model *decoder-only* and what makes training
efficient, since each position's prediction is correctly conditioned on
only its own prefix. See section 6.

The property is verified directly: `test/warpweft/model_test.exs` mutates
tokens after position `t` and asserts the logits at positions up to `t` do
not move, for both positional schemes. If that test fails, the model is
seeing its own answers.

### Multiple heads

The 256-dimensional vector is split into 4 heads of 64 dimensions, each
performing the above independently, and the results are concatenated. One
head can express one weighted average, so one relationship at a time.
Splitting lets the model track several relationships in parallel at the
same cost, since `4 x 64 = 256` either way.

Heads specialise measurably. In the trained Shakespeare model, six of the
eight heads in layers 2 and 3 place most of their attention exactly one
position back, while layer 0 and 1 heads spread broadly. See
`mix wf.attention`.

---

## 6. Why every position predicts at once

This deserves its own section because it is easy to miss and it explains
the shape of everything else.

A `{B, T, vocab}` output means the model emits `T` predictions per
sequence, not one. Because of the causal mask, the prediction at position
`i` depends only on tokens `0..i`, so it is exactly the prediction the
model would make if you had fed it only that prefix.

One forward pass over a 128-token window therefore yields **128 correctly
conditioned training examples**. Training computes the loss at every
position simultaneously:

```
logits {32, 128, 4096}  ->  reshape  {4096, 4096}
targets {32, 128}       ->  reshape  {4096}
cross-entropy over all 4,096 predictions at once
```

Without the mask you would need a separate pass per position, and training
would be roughly `T` times more expensive.

At generation time the same property becomes a nuisance rather than a
benefit: you only want the last position's prediction, and computing the
other 127 is waste. That is what the KV cache addresses, in section 10.

---

## 7. The feed-forward network

Attention moves information *between* positions. It does no per-position
processing to speak of, since its output is a weighted average of values.
The feed-forward network is where per-position computation happens. It is
applied identically and independently at every position.

Two variants, selectable with `mlp:`.

### GELU (`mlp: :gelu`)

```
fc.kernel   : {256, 1024}
proj.kernel : {1024, 256}
h = gelu(x · W_fc)
out = h · W_proj
```

Expand fourfold, apply a non-linearity, project back.

### SwiGLU (`mlp: :swiglu`, the default)

```
w1.kernel : {256, 704}      gate
w3.kernel : {256, 704}      value
w2.kernel : {704, 256}      projection
out = (silu(x · W1) * (x · W3)) · W2
```

Three matrices instead of two. One branch produces a value, the other
produces a **gate** which multiplies it elementwise, so the network learns
per-channel how much of each signal to let through. The hidden size is
`8/3 * d_model` rounded up to a multiple of 32, which keeps the parameter
count comparable to the fourfold GELU version.

SwiGLU measured about 0.11 nats better, and trained about 37% faster here
because our GELU uses the exact `erf` formulation, which is expensive on
CPU.

### Where the parameters live

```
mlp  540,672 per block     12.7% of the model, per block
attn 262,144 per block      6.2% of the model, per block
```

The feed-forward layers hold roughly twice the parameters of attention.
This surprises people who think of the transformer as "the attention
architecture". Attention is the interesting part conceptually; the MLP is
where most of the capacity sits.

---

## 8. The residual stream and normalisation

### Residual connections

Each sub-layer adds to its input rather than replacing it:

```
x = x + attention(norm(x))
x = x + feed_forward(norm(x))
```

Two consequences. Gradients flow directly back along the addition path,
which is what makes deep stacks trainable. And each block only has to
learn a *refinement*, since whatever earlier blocks contributed is still
present.

The useful mental model is a shared communication channel running the
length of the network. Blocks read from it, compute, and write their
contribution back. Nothing is ever overwritten.

### Normalisation

Activations in a deep stack drift towards very large or very small
magnitudes. Normalisation rescales each position's vector back to a
consistent size.

```
LayerNorm : gamma * (x - mean) / sqrt(var + eps) + beta      2 * 256 params
RMSNorm   : gamma * x / sqrt(mean(x^2) + eps)                    256 params
```

RMSNorm skips the mean subtraction, so it rescales without recentring. We
measured **no meaningful difference** between them, under 0.01 nats, which
is worth knowing given how much discussion the choice attracts. RMSNorm is
the default because it is simpler and has half the parameters.

### Pre-norm

The norm sits *inside* the residual branch, before each sub-layer, rather
than after the addition:

```
x = x + attention(norm(x))     pre-norm, what we do
x = norm(x + attention(x))     post-norm, the original 2017 arrangement
```

Pre-norm leaves the residual stream itself unnormalised, giving gradients a
clean path from the loss all the way to the embedding. It is a large part
of why deep transformers became trainable without elaborate warmup
schedules.

---

## 9. The output head

Project from `d_model` back to one score per vocabulary entry.

```
untied : lm_head.kernel {256, 4096}    1,048,576 extra parameters
tied   : reuse wte transposed          0 extra parameters
```

**Weight tying** reuses the embedding matrix. The justification is that
"the vector representing token `t`" and "the direction that indicates
token `t`" describe the same relationship from two sides. It saves a
million parameters here, a quarter of the model.

The output is **logits**: unbounded real numbers, one per vocabulary
entry. They become probabilities only when softmax is applied, which
happens inside the loss function during training and inside sampling
during generation, never in the model itself.

---

## 10. Generation and the KV cache

At generation time the model runs once per token, and the naive approach
recomputes everything each time.

The observation that fixes it: because of the causal mask, position `p`'s
key and value depend only on tokens `0..p`. **They cannot change when
later tokens arrive.** So compute them once and store them.

```
cache per layer:  k {1, n_head, block_size, head_dim}
                  v {1, n_head, block_size, head_dim}
```

Each new token then computes only its own query, key and value, writes the
key and value into the cache at its position, and attends against the
whole cache:

```
q      {1, 4, 1, 64}
keys   {1, 4, 128, 64}    from the cache
scores {1, 4, 1, 128}     one row instead of a 128 x 128 grid
```

Per-token cost drops from O(context²) to O(context). Measured: 7.21 ms to
1.06 ms per token, with byte-identical output.

Two structural consequences:

- **Keys are cached after RoPE rotation.** Rotating on write is what makes
  the cached scores exactly equal to the full forward pass.
- **The cache cannot slide its window.** Cached keys were encoded at their
  original absolute positions, so shifting them along corrupts the
  positional information for RoPE and learned embeddings alike. Once the
  context is full, this implementation falls back to recomputing. Real
  systems solve this with relative-position schemes or explicit eviction
  policies. It is a genuine architectural tension, not an implementation
  shortcut.

Implementation: `lib/warpweft/model/decode.ex`, verified against the full
forward pass for all eight variant combinations.

---

## 11. The variant matrix

Four independent axes, all resolved when the function is traced, so each
combination compiles to its own specialised program with no runtime
branching.

| Axis | Options | Default | Measured effect |
|---|---|---|---|
| `pos` | `:rope`, `:learned` | `:rope` | RoPE better by ~0.19 nats |
| `norm` | `:rms_norm`, `:layer_norm` | `:rms_norm` | no difference beyond noise |
| `mlp` | `:swiglu`, `:gelu` | `:swiglu` | SwiGLU better by ~0.11 nats, and faster |
| `tie_embeddings` | `true`, `false` | `true` | saves 1M parameters |

Run any combination:

```sh
mix wf.train --pos learned --norm layer --mlp gelu --no-tie
```

The important caveat on all four numbers: they were measured on a 3.5M
parameter model trained for 750 steps on 1.1 MB of Shakespeare. They are
evidence about this scale, not universal laws. The one result that held
much larger was the non-architectural one: **twelve times more data
mattered roughly an order of magnitude more than the best architectural
change.**

---

## 12. What is deliberately absent

Worth knowing, because these appear in most production transformers and
their absence keeps this implementation readable.

- **Encoder and cross-attention.** This is decoder-only, the GPT family
  shape. Translation-style models add an encoder stack and attention from
  decoder to encoder.
- **Biases.** Every linear layer here is bias-free, which is standard in
  modern implementations and costs nothing measurable.
- **Flash attention.** The `T x T` score matrix is materialised in full.
  Flash attention computes it in tiles that fit in fast memory, which
  matters enormously on GPUs and is irrelevant at our scale.
- **Grouped-query or multi-query attention.** Sharing keys and values
  across heads to shrink the KV cache. Only matters when the cache
  dominates memory.
- **Mixture of experts.** Routing each token to a subset of several
  feed-forward networks.
- **Dropout at inference.** Present during training, traced away entirely
  at inference rather than being scaled at runtime.

---

## 13. Reading the code

In dependency order, shortest first:

| File | Contents |
|---|---|
| `lib/warpweft/model/layers.ex` | norms, activations, the two MLP variants |
| `lib/warpweft/model/rope.ex` | rotary tables and their application |
| `lib/warpweft/model/attention.ex` | causal self-attention, and the cached variant |
| `lib/warpweft/model.ex` | parameter initialisation, the block loop, the head |
| `lib/warpweft/model/decode.ex` | single-position forward pass for generation |
| `lib/warpweft/config.ex` | every dimension and variant in one struct |

Then run `mix wf.explain` to see one prompt traverse all of it with real
numbers, and `mix wf.attention --heatmaps` to see what the heads learned.
