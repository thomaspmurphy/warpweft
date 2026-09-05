# Concepts

A reference for every concept this project uses. Each entry gives the
definition, why it matters, and where it appears here.

Grouped by area rather than alphabetically, because the groupings are part
of the explanation. For how the pieces fit together structurally, see
[ARCHITECTURE.md](ARCHITECTURE.md). For the engineering practices, see
[TECHNIQUES.md](TECHNIQUES.md).

- [Language modelling](#language-modelling)
- [Tokenisation](#tokenisation)
- [Tensors and numerical computing](#tensors-and-numerical-computing)
- [Model components](#model-components)
- [Attention](#attention)
- [Training](#training)
- [Measuring quality](#measuring-quality)
- [Generation](#generation)
- [Interpretability](#interpretability)

---

## Language modelling

**Language model.** A function mapping a sequence of tokens to a
probability distribution over what comes next. That single capability,
applied repeatedly, produces text.

**Autoregressive.** Generating one element at a time, each conditioned on
everything produced so far. The output of step `n` becomes part of the
input to step `n + 1`.

**Decoder-only.** An architecture with a single stack of layers where each
position sees only earlier positions. The GPT family shape. Contrast with
encoder-decoder models, which read a whole input with unrestricted
attention and then generate an output.

**Next-token prediction.** The training objective. Its appeal is that it
requires no labelling: the correct answer is simply the next token in the
text, so any text is training data.

**Context window.** How many tokens the model can attend over, called
`block_size` here. Ours is 128 or 256. Cost grows with the square of this
number, because attention compares every position against every other.

**Prompt.** The tokens given to the model before it starts generating.
Mechanically there is no difference between prompt tokens and generated
ones; the prompt is just the part you supplied.

---

## Tokenisation

**Token.** The atomic unit the model operates on. Not a word. In our
TinyStories tokenizer, `" chair"` is one token and an unusual word may be
three.

**Vocabulary.** The complete set of tokens, each with an integer id. Ours
are 1024 and 4096. The final layer produces one score per vocabulary
entry, so vocabulary size directly drives the cost of that layer.

**Byte-level.** Using the 256 possible byte values as the base alphabet.
The consequence is that any input whatsoever can be represented, including
emoji, other scripts and corrupt data. There is no "unknown token" and
`decode(encode(x)) == x` always holds.

**Byte Pair Encoding (BPE).** A compression algorithm repurposed for
tokenisation. Start from the byte alphabet, repeatedly find the most
frequent adjacent pair of tokens and merge it into a new token. Frequent
sequences such as `the` and `ing` become single tokens; rare words remain
decomposed.

**Merge.** One learned rule of the form "when token A is followed by token
B, replace the pair with new token C". A tokenizer *is* an ordered list of
merges; everything else derives from it. Ours learns 768 merges for
Shakespeare.

**Pre-tokenisation.** Splitting text into words, numbers, punctuation runs
and whitespace *before* BPE runs, so merges never learn across those
boundaries. Without it the tokenizer would learn tokens like `dog.The`,
which are artefacts of formatting rather than units of language.

**Special token.** An entry outside the learned vocabulary, used as a
marker rather than as text. We use `<|endoftext|>` to separate documents,
which lets the model learn where stories end and gives generation a
natural place to stop.

**Compression ratio.** Bytes per token. Ours are 2.44 for Shakespeare at
vocab 1024, 3.97 for TinyStories at vocab 4096. Higher means each forward
pass covers more text, but also that a fixed corpus yields fewer training
examples, which matters when data is scarce.

---

## Tensors and numerical computing

**Tensor.** A multi-dimensional array with a fixed shape and element type.
Everything the model touches is a tensor.

**Shape.** The size along each dimension, written `{32, 128, 256}` for a
batch of 32 sequences, 128 positions each, 256 numbers per position.
Getting shapes right is most of the work of implementing a model.

**Broadcasting.** Automatically expanding a smaller tensor to match a
larger one in an elementwise operation, so a `{256}` vector can be added
to every row of a `{32, 128, 256}` tensor without materialising copies.

**Dot product.** Multiply two vectors elementwise and sum. It measures
alignment: large when vectors point the same way. Attention is built on
this.

**Matrix multiplication.** Many dot products at once, and the operation
that dominates the compute in a transformer.

**Gather.** Selecting elements by index, `Nx.take` here. Used for
embedding lookup (fetch the vector for each token id) and for batch
sampling (fetch many windows from a corpus in one operation).

**Element type.** We use 32-bit floats throughout for parameters and
activations, and 32-bit integers for token ids. Larger models often use
16-bit floats to halve memory traffic.

**JIT compilation.** Compiling numerical code to native machine code just
before running it. `Nx.Defn.jit` traces a function, builds a computation
graph and hands it to XLA.

**XLA.** The compiler underneath, which fuses many small operations into
few optimised kernels. Its central constraint for us: it recompiles
whenever tensor **shapes** change, which is why keeping shapes constant is
a first-order performance concern.

**Tracing.** Running a function with placeholder tensors to record the
operations it performs. Anything decided by ordinary Elixir control flow
during tracing is baked into the compiled program, which is how our
architecture variants cost nothing at runtime.

**Device and host.** The device is where tensors live for computation; the
host is the Elixir process. Moving data between them is expensive, which
is why the batch pipeline never brings data back to the host.

---

## Model components

**Parameter.** A number learned during training. Ours has 4,262,144 of
them. "Model size" means this count.

**Embedding.** A lookup table with one learned vector per vocabulary
entry, converting integer ids into vectors. Related tokens end up with
similar vectors, which is learned rather than designed.

**`d_model`.** The width of the residual stream, 256 here. Every position
is represented by this many numbers throughout the network.

**Block (or layer).** One repeat of attention plus feed-forward. Ours has
four. Depth lets later blocks build on the representations earlier ones
produced.

**Residual stream.** The running representation that blocks add into,
rather than replacing. Written `x = x + sublayer(norm(x))`. It gives
gradients a direct path to every layer and lets each block learn a
refinement rather than a whole representation.

**Residual connection (skip connection).** The addition itself. The single
most important trick for making deep networks trainable.

**Feed-forward network (MLP).** A two- or three-matrix network applied
independently at each position. Where per-position processing happens, and
where roughly two-thirds of the parameters live.

**Activation function.** The non-linearity between matrix multiplies.
Without one, stacked layers collapse mathematically into a single matrix
and depth buys nothing.

**GELU.** A smooth approximation of "keep positives, discard negatives".
Standard in the original GPT models. The exact form uses `erf` and is
slow on CPU.

**SiLU (swish).** `x * sigmoid(x)`. Another smooth gate-like
non-linearity, used inside SwiGLU.

**SwiGLU.** A gated feed-forward variant: one branch produces values,
another produces a gate that multiplies them elementwise, so the network
learns per-channel how much signal to pass. Measured better and faster
than GELU here.

**Normalisation.** Rescaling each position's vector to a consistent
magnitude, preventing values from drifting to extremes across a deep
stack.

**LayerNorm.** Subtract the mean, divide by the standard deviation, then
apply a learned scale and shift.

**RMSNorm.** Divide by the root mean square, with a learned scale. Skips
the mean subtraction, so it rescales without recentring. Half the
parameters, and no measurable quality difference at our scale.

**Pre-norm and post-norm.** Whether normalisation is applied inside the
residual branch before the sub-layer (pre-norm, what we do) or after the
addition (post-norm, the original 2017 design). Pre-norm keeps the
residual path clean and is much easier to train deep.

**Logits.** The raw, unbounded scores the model outputs, one per
vocabulary entry. They become probabilities only when softmax is applied.

**Softmax.** Converts a vector of scores into a probability distribution
by exponentiating and dividing by the total. Everything becomes positive
and sums to 1.

**Weight tying.** Reusing the embedding matrix, transposed, as the output
projection. Saves a quarter of our parameters, on the reasoning that
"the vector for token `t`" and "the direction indicating token `t`" are
two views of one relationship.

---

## Attention

**Attention.** A mechanism for each position to pull in information from
other positions, with the weighting decided by content rather than fixed
in advance. The defining component of the architecture.

**Query, key, value.** Three vectors each position produces by
multiplication with learned matrices. The query is what this position is
looking for, the key is what it offers to others, the value is what it
passes on when selected. Think of a dictionary lookup where every key
matches partially and you get a weighted blend of all the values.

**Attention weights.** The `{batch, head, query, key}` tensor of how much
each position attends to each other position. Every query row is a
probability distribution summing to 1.

**Scaled dot-product attention.** The standard formulation:
`softmax(Q · Kᵀ / sqrt(head_dim)) · V`. The scaling factor keeps score
magnitudes stable as dimension grows, without which softmax saturates and
gradients vanish.

**Attention head.** One independent attention computation over a slice of
the vector. One head can express one weighted average, so one relationship
at a time.

**Multi-head attention.** Several heads in parallel over disjoint slices,
concatenated. Same total cost, but several relationships can be tracked at
once.

**`head_dim`.** Dimension per head, `d_model / n_head`, so 64 here.

**Causal mask.** Forcing attention weights to zero for future positions,
by setting their scores to negative infinity before softmax. What makes
the model *decoder-only*, and what lets one forward pass produce a correct
prediction at every position simultaneously.

**Positional encoding.** Information about word order, which must be
supplied explicitly because attention is permutation-invariant on its own.

**Learned positional embeddings.** A lookup table with one learned vector
per position, added to the token embedding.

**Rotary embeddings (RoPE).** Rotating query and key vectors by an angle
set by their position, so that their dot product depends only on the
distance between them. Costs no parameters and encodes relative rather
than absolute position.

**KV cache.** Storing each position's keys and values during generation so
they are computed once rather than recomputed for every subsequent token.
Valid precisely because the causal mask means those values cannot change.
Reduces per-token cost from quadratic to linear in context length.

---

## Training

**Gradient.** For each parameter, the direction and magnitude of change
that would most increase the loss. Training moves parameters in the
opposite direction.

**Backpropagation.** Computing all gradients efficiently by applying the
chain rule backwards through the network.

**Automatic differentiation.** The framework deriving gradients from the
forward code, so nobody writes derivatives by hand. `Nx.Defn.value_and_grad`
here, and the one substantial piece of machinery we did not implement
ourselves.

**Loss function.** A single number measuring how wrong the model is, which
training minimises.

**Cross-entropy.** The standard loss for classification: the negative log
of the probability assigned to the correct answer. Confident and correct
gives near zero; confident and wrong is heavily penalised.

**Optimiser.** The rule converting gradients into parameter updates.

**SGD.** The simplest optimiser: subtract the gradient times a learning
rate.

**Adam and AdamW.** Optimisers that track a running estimate of each
parameter's typical gradient magnitude and normalise by it, so parameters
with consistently small gradients still move. AdamW additionally applies
weight decay correctly, separated from the gradient.

**Learning rate.** How large a step to take. The most important
hyperparameter: too large diverges, too small never arrives.

**Learning-rate schedule.** Varying the rate during training. Ours ramps
up linearly over 200 steps then decays along a cosine curve.

**Warmup.** The initial ramp. A large step into a randomly initialised
network can destroy it before it learns anything, so the rate starts near
zero.

**Cosine decay.** Smoothly reducing the rate towards a floor, so the model
settles into a minimum rather than bouncing around it.

**Gradient clipping.** Scaling gradients down when their total magnitude
exceeds a threshold. Cheap insurance against a single unlucky batch
destroying hours of training.

**Weight decay.** Gently pulling parameters towards zero each step, a
regularisation that discourages the model from relying on any single large
weight.

**Batch.** Several sequences processed together. Ours is 32 sequences of
128 tokens, so 4,096 predictions per step.

**Step (iteration).** One batch: forward, gradient, update.

**Epoch.** One full pass over the training data. Less meaningful here,
since we sample random windows rather than iterating in order.

**Dropout.** Randomly zeroing a fraction of activations during training,
which prevents the model depending too heavily on any single pathway.
Disabled at inference.

**Regularisation.** Any technique that trades training accuracy for
generalisation. Dropout, weight decay and early stopping are all
regularisation.

**Overfitting.** Learning the training data specifically rather than the
pattern behind it. Visible as training loss falling while validation loss
rises.

**Generalisation gap.** The difference between training and validation
loss. Ours went from 0.754 on Shakespeare to 0.139 on TinyStories, with an
identical model and twelve times the data.

**Train/validation split.** Holding back some data to measure on. Our
split is 90/10, cut at a document boundary where one exists so no document
appears in both halves.

**Early stopping.** Keeping the checkpoint with the best validation loss
rather than the last one. We do this automatically via `best.ckpt`, which
is why the deployed Shakespeare model is not the overfit end-of-run one.

**Checkpoint.** Saved parameters, optionally with optimiser state and step
count so training can resume exactly.

**Initialisation.** The random starting values. Ours uses a normal
distribution with standard deviation 0.02, with residual output
projections scaled down by `1/sqrt(2 * n_layer)` to keep the residual
stream's variance stable as depth grows.

**Tokens per parameter.** Training tokens divided by parameter count. A
rough health indicator, with roughly 20 often quoted as compute-optimal.
Ours were 0.118 (badly starved) and 1.198 (workable).

**Scaling laws.** The empirical relationships between model size, data
volume, compute and loss. The practical takeaway we ran into directly:
model and data must be scaled together, and getting the ratio wrong wastes
whichever one is in surplus.

---

## Measuring quality

**Nats.** Loss measured with natural logarithms. A loss of `ln(n)` means
the model is as uncertain as if choosing uniformly among `n` options, so
2.0 nats corresponds to about 7.4 equally likely choices.

**Bits.** The same thing in base 2. Divide nats by `ln(2)`.

**Perplexity.** `exp(loss)`, the effective number of equally likely
choices. Another presentation of the same quantity.

**Bits per byte.** Loss normalised by how much text each token covers. The
**only** fair way to compare models with different tokenizers, because a
tokenizer packing more text per token earns a higher per-token loss for
identical quality. Ours are 2.036 and 0.732.

**Baseline.** A simple reference to calibrate against. We compare with a
uniform distribution, unigram and bigram models, and `gzip` and `xz`,
which puts our 0.732 bits per byte in context rather than leaving it as a
bare number.

**Language modelling as compression.** A model assigning high probability
to real text can be used to compress that text, and the loss in bits per
byte *is* the compression rate achievable. Our model beats `xz -9` by more
than a factor of two, which is a concrete statement of what it has
learned.

---

## Generation

**Sampling.** Drawing a token from the predicted distribution rather than
always taking the most likely one, which would produce flat, repetitive
text.

**Greedy decoding.** Always taking the highest-scoring token. Equivalent
to top-k of 1.

**Temperature.** Dividing logits before softmax. Below 1 sharpens the
distribution towards the favourite, above 1 flattens it. Note that zero is
not "greedy" but a division by zero.

**Top-k sampling.** Discarding all but the `k` highest-scoring tokens
before sampling, so the long tail of thousands of individually unlikely
tokens cannot collectively steal probability.

**Top-p (nucleus) sampling.** Keeping the smallest set of tokens whose
probabilities sum to `p`, adapting the cutoff to how confident the model
is. Not implemented here, but the natural next step from top-k.

**Gumbel-max trick.** Adding Gumbel-distributed noise to logits and taking
the argmax gives an exact sample from the softmax distribution. One
vectorised operation with no loops, no cumulative sums and no branching,
which lets sampling compile into the same program as the forward pass.

**Prefill.** Running the prompt through the model to populate the KV cache
before generation begins.

**Decode step.** Producing one token: one position forward, sample,
append.

---

## Interpretability

**Attention pattern.** The grid of weights for one head, showing which
positions attend to which. The most directly inspectable thing inside a
transformer.

**Previous-token head.** A head concentrating its attention one position
back. Six of the eight heads in layers 2 and 3 of our Shakespeare model
behave this way, and none were designed to.

**Induction head.** A head attending from the current token to whatever
followed its *previous occurrence*, which is the circuit that copies a
name or phrase forward. Our model shows no sign of them, which explains
its consistent entity-tracking failures: `"Buzzy"` becomes `"Bobo"`
halfway through a story.

**Attention sink.** A head dumping probability on position 0 when it has
nothing useful to contribute, because softmax forces every row to sum to 1
and it must put the mass somewhere.

**Attention entropy.** How spread out a head's attention is. Low means a
sharp lookup at one position, high means a diffuse average.

**Attention distance.** How far back a head looks on average.

**Null baseline.** What a statistic reads under the *absence* of the
behaviour being measured. Essential here: position 0 is visible to every
query row while the last position is visible to one, so a perfectly
uniform head puts `H(t)/t` of its mass on position 0 and looks like a
sink. Reporting sink mass as a multiple of that baseline removed half our
apparent sinks.
