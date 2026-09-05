# Techniques

The engineering techniques used to build this, organised by kind. Most of
them have nothing to do with machine learning and transfer directly to
ordinary software.

Each entry says what the technique is, where it appears here, and what it
is worth. For the vocabulary see [CONCEPTS.md](CONCEPTS.md); for the model
structure see [ARCHITECTURE.md](ARCHITECTURE.md).

- [Parsing and text processing](#parsing-and-text-processing)
- [Numerical and array programming](#numerical-and-array-programming)
- [Compilation and performance](#compilation-and-performance)
- [Functional programming in Elixir](#functional-programming-in-elixir)
- [Data engineering](#data-engineering)
- [Testing](#testing)
- [Measurement and benchmarking](#measurement-and-benchmarking)
- [API and tooling design](#api-and-tooling-design)

---

## Parsing and text processing

### A regex as an executable specification

The tokenizer splits text with a regex. We later replaced that with a
hand-written scanner about seven times faster, but the regex stayed in the
test suite as the **definition of correct**. A test asserts the fast
implementation agrees with it byte for byte.

This is worth generalising: when you optimise something, keep the slow
obvious version and test the fast one against it. Ours earned its keep
twice. It caught an off-by-one on the second chunk of the corpus, and much
later it caught a genuine misunderstanding of what `\s` matches in Elixir
regexes. See `test/warpweft/tokenizer/bpe_test.exs`.

### Byte-level parsing with a fast path and an escape hatch

The scanner handles ASCII with direct byte comparisons and defers to the
regex for anything else, keeping Unicode letter and number categories
exact without paying for them on the 99% of text that is ASCII.

The subtlety that cost us a regression: the escape must be **fine-grained**.
Our first version bailed out for a whole 64 KB slice on encountering any
non-ASCII byte, which was slower than not optimising at all, because 0.06%
of TinyStories bytes are non-ASCII but they are scattered across 88% of
slices. Escaping per-run instead turned a 1.16x regression into a 3.8x
win.

The general rule for fast-path designs: measure **how often the fast path
is denied**, not how rare the slow case is. Those differ by orders of
magnitude.

### Binary pattern matching

Elixir matches on binaries directly, which compiles to efficient byte
inspection without copying:

```elixir
defp chunk(<<?\s, c, rest::binary>>, acc) when c not in @whitespace
```

Combined with index arithmetic over `:binary.at/2` and `:binary.part/3`,
this gave the roughly sevenfold speedup over the regex. It is also where
an index-versus-length off-by-one crept in, which is exactly why the
differential test exists.

### Safe split points

To process a large binary in slices, you must cut where no token can
straddle the boundary. Ours cuts at a whitespace byte preceded by a
non-whitespace byte, and requires **both** to be ASCII, because a non-ASCII
byte might be Unicode whitespace we cannot recognise byte-wise.

Reasoning explicitly about which positions are safe, rather than cutting
at a fixed offset and hoping, is the difference between a correct
optimisation and a subtle corruption.

### Incremental UTF-8 decoding

Streaming byte-level tokens to a terminal needs care, because one token
can end mid-character. The emitter holds back trailing bytes that could
still be completed, and distinguishes two cases:

- **Incomplete**: a valid prefix of a longer character. Hold it.
- **Malformed**: bytes that can never form a character. Substitute U+FFFD
  and continue, or the stream stalls forever waiting.

Getting only the first case right still emits invalid UTF-8, which a test
caught immediately. See `emitter/1` in `lib/warpweft/generate.ex`.

### Memoisation over a natural key

Encoding a corpus applies merge rules per chunk. Since chunks repeat
enormously (Shakespeare has 294,763 chunks but only 15,057 distinct ones),
memoising on the chunk string makes the cost proportional to *unique*
chunks rather than total ones.

### Frequency-table algorithms

Training BPE naively rescans the corpus for every merge. Instead we build
a frequency table of unique chunks once and merge within that table, so
each of 768 merges touches 15,057 entries rather than a million-character
corpus. This is what makes pure-Elixir tokenizer training take six seconds
rather than hours.

---

## Numerical and array programming

### Vectorisation instead of loops

Nearly every loop you would write imperatively becomes one array
operation. Sampling a batch is the clearest case. Rather than looping to
extract windows:

```elixir
offsets = random integers            {batch}
indices = offsets[:, newaxis] + iota({1, block + 1})
windows = Nx.take(corpus, indices)   {batch, block + 1}
```

Broadcasting a column of offsets against a row of positions produces every
index needed, and one gather fetches all of them. No loop, no host
round-trip.

### Masking instead of branching

Array code cannot branch per element, so conditionals become arithmetic.
The causal mask is the canonical example: rather than skipping future
positions, add negative infinity to their scores so softmax assigns them
zero.

```elixir
mask = Nx.greater_equal(iota_rows, iota_cols)
scores = Nx.select(mask, scores, -1.0e9)
```

Top-k filtering uses the same idea: instead of removing tokens, set the
losers to negative infinity.

### Building index tensors from iota

`Nx.iota` generates coordinate tensors, and comparing two of them along
different axes produces structured masks without loops or literals. The
causal mask, the diagonal offsets used in attention statistics, and the
position indices for RoPE are all built this way.

### Functional randomness with explicit keys

There is no global random state. A PRNG key is a value, split explicitly
whenever two independent streams are needed:

```elixir
{key1, key2} = Nx.Random.split(key)
```

Verbose, but it makes randomness reproducible and traceable. Our original
dropout implementation reused one constant key and therefore applied the
*same mask every step*, which is silent and would be nearly impossible to
notice without knowing to look.

### Numerical stability

Two habits worth carrying:

- **Subtract the maximum before exponentiating** in softmax, so `exp` never
  overflows. Mathematically a no-op, practically essential.
- **Scale dot products by `1/sqrt(dim)`**, or softmax saturates as
  dimension grows and gradients vanish.

### Know what silently clamps

`Nx.take` clamps out-of-range indices rather than raising. That makes
bounds assertions on gather indices vacuous, and lets a sampler running
off the end of a corpus produce windows whose tails are one repeated
token, with nothing anywhere reporting a problem. Worth knowing which
operations in your array library fail loudly and which do not.

---

## Compilation and performance

### Fixed shapes so the compiler works once

XLA recompiles whenever tensor shapes change. Our three largest speedups
all came from shape stability rather than better mathematics.

The starkest: generating with a growing sequence recompiles on every
token, at 99.58 ms each. Padding the context to a fixed-size buffer and
tracking the length as a scalar compiles once, at 7.21 ms each. Identical
arithmetic, fourteen times faster.

When a compiler or cache sits in your hot path, **stability of whatever it
keys on** is a first-order design concern.

### Trace-time versus runtime values

Values known while tracing get baked into the compiled program. This is
what makes the architecture variants free:

```elixir
case cfg.norm do
  :rms_norm -> rms_norm(x, params)
  :layer_norm -> layer_norm(x, params)
end
```

`cfg.norm` is read during tracing, so each variant compiles to its own
specialised program with no runtime branch. The cost is that each
combination compiles separately.

The same distinction guided an API change: temperature became a runtime
tensor argument so the REPL can change it freely, while top-k must stay
trace-time because it determines an output shape.

### Fuse the whole step into one program

The training step traces forward pass, gradient, gradient clipping and the
optimiser update into a single compiled function. Generation fuses the
forward pass, temperature, top-k masking and sampling. Fewer, larger
programs give the compiler more to work with and cut per-call overhead.

### Caching a computation that provably cannot change

The KV cache is a specific instance of a general move: identify work whose
inputs cannot have changed, and store it. The causal mask guarantees
position `p`'s key and value depend only on tokens up to `p`, so they are
fixed once written.

The general lesson is that the *justification* matters as much as the
cache. Ours came with a constraint attached, since the cached values
encode absolute positions and therefore cannot be shifted, which is why
the window cannot slide.

### Avoiding host round-trips

Every transfer between device memory and the Elixir process costs. The
original batch sampler pulled random indices to the host, built windows
with `Enum.map`, and pushed them back, once per step. Doing it entirely on
device removed that per-step cost.

### Streaming to control memory pressure

Processing a 22 MB corpus by materialising all 5.5 million chunks in one
list is slower than processing it in slices, purely from garbage
collection. Scanning the same bytes document by document took 1,814 ms
against 2,679 ms for one pass. Same work, 1.5x faster, from letting
intermediate values die young.

---

## Functional programming in Elixir

### Pattern matching for dispatch

Multi-clause functions replace conditionals throughout, especially over
binary structure and over the `{:ok, _}` / `{:error, _}` shapes returned
by IO.

### Reduce as the universal loop

`Enum.reduce` for the block stack and the training loop,
`Enum.reduce_while` where early exit is needed (stopping on an
end-of-text token or a context limit). Threading state explicitly through
a reduction keeps it visible rather than hidden in mutable variables.

### Streams for unbounded sequences

The training data is an infinite stream of random batches built with
`Stream.unfold`, carrying the PRNG key as its state. The consumer takes as
many as it wants. Generation and evaluation compose over the same stream.

### Closures capturing configuration

`Nx.Defn.jit` receives a closure that has captured the config, so the
config is available at trace time without being a tensor argument:

```elixir
Nx.Defn.jit(fn params, x, y -> ... uses cfg ... end)
```

### Data as plain maps

Parameters are a nested map of tensors, not an opaque framework object.
You can inspect them, walk them, count them and serialise them with
ordinary tools. For a learning project this is the whole point, and it is
also what made weight tying and the KV cache straightforward.

### Iodata for string building

Building output as nested lists of binaries and flattening once with
`IO.iodata_to_binary` avoids repeated concatenation.

---

## Data engineering

### Precompute once, load fast

Tokenising happens once, offline, writing token ids to a flat binary file
of unsigned 16-bit integers. Training loads the whole corpus with one
`File.read!` and one `Nx.from_binary`, and never touches text again. The
training loop contains no string processing at all.

### Choose the narrowest type that fits

Token ids fit in `u16` for any vocabulary up to 65,535, halving the file
size against `u32`. They are widened to `s32` once at load, because narrow
integer types can behave awkwardly in gather operations.

### Split on semantic boundaries

The train/validation split cuts at a document boundary where the corpus
has one, so no document appears in both halves. Without that, validation
would be measuring partly-memorised continuations of training documents.

Guard the degenerate case: our first version would happily produce a
1%/99% split if the only boundary sat near the start, so it now falls back
to the target position when the nearest boundary is too far away.

### Self-contained artefacts

Each training run writes a directory holding its config, a pointer to its
tokenizer, and its checkpoints. Anything that loads a run gets a
consistent set, and there is no global state to get out of sync.

### Name outputs after what happened, not what was requested

BPE stops merging early when a corpus runs out of pairs, so asking for
4096 tokens can yield 3,891. Naming the tokenizer directory after the
requested number while naming the tokenized data after the achieved number
produced a pipeline that failed on a file no code path ever wrote. Both
now use the achieved value.

---

## Testing

### Property-based testing

Some properties should hold for *all* inputs, and generated inputs find
counterexamples humans do not think of:

```elixir
property "decode(encode(s)) == s for arbitrary binaries" do
  check all s <- StreamData.binary(max_length: 200) do
    assert BPE.decode(bpe, BPE.encode(bpe, s)) == s
  end
end
```

Best suited to round trips, invariants and algebraic laws.

### Differential testing

Assert a new implementation matches a reference one across many inputs.
Used for the tokenizer against its regex specification, and for the KV
cache against the full forward pass. This is the strongest tool available
when replacing something with a faster equivalent.

### Mutation testing

**The single highest-value technique in this project.** A passing suite
tells you the tests pass, not that they test anything. Deliberately break
the implementation and check the suite notices.

Three things here turned out to be tested only in appearance: a bounds
check that could not fail because the underlying operation clamps; the
learning-rate schedule and weight decay, both removable with all 59 tests
still green; and temperature, which could be deleted entirely with no
failures. All three looked like real coverage.

If you adopt one practice from this repository, adopt this one. Pick your
three most important tests, break the code they cover, and confirm they go
red.

### Testing invariants rather than outputs

The most valuable tests here assert structural properties, not specific
values:

- **Causal masking**: change tokens after position `t`, and logits at `t`
  must not move. If this fails the model is seeing its own answers and
  every result is void.
- **Padding safety**: logits must be identical under zero or garbage
  padding, which is what licenses the fixed-shape generation buffer.
- **RoPE relative invariance**: the score between a rotated query and key
  must depend only on the distance between positions.

Each states something the design *relies* on, so each fails loudly if a
refactor breaks an assumption.

### Equivalence testing between paths

Where two implementations must agree, test that directly across a matrix
of settings. The cached and recomputing generation paths are compared
across seeds, temperatures and top-k values, and must produce
byte-identical text.

### Checked-in fixtures over conditional skips

A test guarded by `if File.exists?(corpus)` passes silently on a clean
checkout while asserting nothing. Checking in a 60 KB sample makes it
always run. A skipped test is worse than a missing one, because it looks
like coverage.

### Overfit a single batch

The cheapest sanity check for any training code: a tiny model on one
batch, and assert the loss goes near zero. If it cannot memorise one
batch, gradients are not flowing and nothing else is worth debugging.

### Write down which mutation a test catches

Where a test guards something subtle, the comment says what breaks without
it. That tells the next reader whether a "simplification" is safe.

---

## Measurement and benchmarking

### Measure in an isolated process

We reported a threefold performance regression that did not exist. The
benchmark had allocated large structures before the measurement, leaving
the garbage collector in a state that made the timing meaningless. One
operation per process fixed it.

On any managed runtime, earlier work in the same process is part of your
experimental setup whether you intended it or not.

### Name the denominator

We measured a component as "3% of the total" and used it to argue against
optimising. It was 3% of one function and 69% of another. A percentage
without a stated scope is not a measurement.

### Probe before committing to a long run

Twice we ran a deliberately tiny version of a long job to find out what the
real one would cost. The second time, a 90-second probe showed a planned
training run would take 8.3 hours, and we redesigned the experiment
instead.

### Calibrate against baselines

A loss of 0.732 bits per byte means nothing alone. Against a uniform
distribution (3.029), `gzip -9` (2.386), a bigram model (1.347) and `xz -9`
(1.729), it means a great deal. Cheap baselines turn an unanchored number
into a claim.

### Work out the null value of a statistic

Before reading a measurement as evidence of a behaviour, work out what it
reads when the behaviour is absent. Our attention-sink metric scored 0.34
on a perfectly uniform head purely from the causal mask's geometry, so
half our apparent "sinks" were an artefact. Ratios against a baseline beat
raw magnitudes whenever the sampling geometry is uneven.

### Change one variable

Our controlled data-volume experiment holds the architecture,
hyperparameters and step count fixed and changes only the corpus. The
tempting alternative would have changed depth, width, context and data
simultaneously and taught us nothing about which mattered.

### Check units before comparing

Cross-entropy in nats per token is not comparable across tokenizers,
because a tokenizer covering more text per token earns a higher loss for
identical quality. Converting to bits per byte divides that out. We
published a comparison in the wrong units before catching it.

---

## API and tooling design

### Error messages that name the fix

Every failure a user can plausibly cause reports what to do:

```
No tokenized data for corpus "shakespeare" at vocab size 9999.
Expected data/tokenized/shakespeare-9999.meta.json.

Prepare it with:
    mix wf.data --corpus shakespeare
    mix wf.tokenizer.train --corpus shakespeare --vocab 9999

Already tokenized at vocab size(s): 1024.
```

That last line, listing what *does* exist, is usually the most useful part
and is cheap to produce.

### Validate at the boundary, not in the depths

`Config.validate!/1` runs at model initialisation, so a bad configuration
fails immediately with a specific message rather than as a reshape error
inside attention. It also catches a case that would otherwise be silent:
an odd head dimension makes RoPE discard a channel with no error at all.

### One resolver, not three copies

Three tasks each had their own copy of "find the newest run", in three
subtly different forms, one of which crashed with a `FunctionClauseError`
when the directory did not exist. Consolidating into `Warpweft.Runs` fixed
all three and added a check none of them had: that the chosen directory
actually contains a checkpoint.

### Make the common path a single command

`mix wf.repl` loads the newest run, warms the compiled sampler and gives
you a prompt. Optional flags cover everything else. The default should be
the thing you want most of the time.

### Build tools that show, not just tell

`mix wf.explain` walks a prompt through every stage with real numbers, and
`mix wf.attention --heatmaps` renders what the heads learned. Both are
worth more for understanding the system than the prose describing it, and
both took under an hour.

### Streaming output for perceived speed

The REPL prints tokens as they are produced rather than waiting for the
full completion. The total time is identical; the experience is not.

### Resumability as a first-class feature

Training checkpoints its parameters, optimiser state and step count, so an
interrupted run continues exactly. This was used in anger when a 20-minute
run was killed at 90%.

The subtlety worth recording: resuming initially reset "best validation
loss so far" to infinity, so the first evaluation after a restart would
overwrite a genuinely better checkpoint. Resumption has to restore *all*
the state that matters, including the bookkeeping.
