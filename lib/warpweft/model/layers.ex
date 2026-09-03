defmodule Warpweft.Model.Layers do
  @moduledoc """
  Hand-written building blocks of the transformer, as plain functions over
  Nx tensors. Everything here is traceable, so the whole forward pass
  JIT-compiles into a single XLA program.

  All linear layers are bias-free (as in most modern GPTs).
  """

  @norm_eps 1.0e-5

  # -- norms -------------------------------------------------------------------

  @doc "LayerNorm over the last axis: gamma * (x - mean) / sqrt(var + eps) + beta."
  def layer_norm(x, %{"gamma" => gamma, "beta" => beta}) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = x |> Nx.subtract(mean) |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

    x
    |> Nx.subtract(mean)
    |> Nx.multiply(Nx.rsqrt(Nx.add(var, @norm_eps)))
    |> Nx.multiply(gamma)
    |> Nx.add(beta)
  end

  @doc "RMSNorm over the last axis: gamma * x / sqrt(mean(x^2) + eps). No re-centering."
  def rms_norm(x, %{"gamma" => gamma}) do
    ms = x |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

    x
    |> Nx.multiply(Nx.rsqrt(Nx.add(ms, @norm_eps)))
    |> Nx.multiply(gamma)
  end

  def norm(x, params, :layer_norm), do: layer_norm(x, params)
  def norm(x, params, :rms_norm), do: rms_norm(x, params)

  # -- activations ---------------------------------------------------------------

  @doc "Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."
  def gelu(x) do
    x
    |> Nx.divide(Nx.sqrt(2.0))
    |> Nx.erf()
    |> Nx.add(1.0)
    |> Nx.multiply(x)
    |> Nx.multiply(0.5)
  end

  @doc "SiLU / swish: x * sigmoid(x)."
  def silu(x), do: Nx.multiply(x, Nx.sigmoid(x))

  @doc "Numerically stable softmax over the last axis."
  def softmax(x) do
    x = Nx.subtract(x, Nx.reduce_max(x, axes: [-1], keep_axes: true))
    e = Nx.exp(x)
    Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
  end

  # -- dropout -------------------------------------------------------------------

  @doc """
  Inverted dropout. `key` is a PRNG key; pass `rate: 0.0` (or trace the
  inference variant of the forward pass) to disable. Scaling by 1/keep_prob
  at train time keeps the inference path a plain identity.
  """
  def dropout(x, _key, rate) when rate == 0.0, do: x

  def dropout(x, key, rate) do
    {mask, _key} = Nx.Random.uniform(key, shape: Nx.shape(x), type: Nx.type(x))
    keep = Nx.greater(mask, rate)
    Nx.select(keep, Nx.divide(x, 1.0 - rate), Nx.tensor(0.0, type: Nx.type(x)))
  end

  # -- mlps ---------------------------------------------------------------------

  @doc """
  Position-wise feed-forward network.

  `:gelu` is the classic GPT MLP `dense(4d) -> GELU -> dense(d)`;
  `:swiglu` is the gated variant `(silu(x W1) * x W3) W2`.
  """
  def mlp(x, %{"fc" => fc, "proj" => proj}, :gelu, key, rate) do
    x
    |> Nx.dot(fc["kernel"])
    |> gelu()
    |> Nx.dot(proj["kernel"])
    |> dropout(key, rate)
  end

  def mlp(x, %{"w1" => w1, "w2" => w2, "w3" => w3}, :swiglu, key, rate) do
    gate = x |> Nx.dot(w1["kernel"]) |> silu()
    up = Nx.dot(x, w3["kernel"])

    gate
    |> Nx.multiply(up)
    |> Nx.dot(w2["kernel"])
    |> dropout(key, rate)
  end
end
