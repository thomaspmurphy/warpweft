defmodule Warpweft.Model.Attention do
  @moduledoc """
  Causal multi-head self-attention.

  One fused computation per block: QKV projection, head split, optional
  RoPE, scaled dot-product with a causal mask, softmax, value mix,
  head merge, output projection.
  """

  alias Warpweft.Model.{Layers, RoPE}

  @doc """
  `x` is `{batch, seq, d_model}`; returns the same shape.

  Options (all trace-time static):
    * `:n_head`  - number of attention heads
    * `:rope`    - `{cos, sin}` tables or `nil` (learned positions upstream)
    * `:dropout` - rate applied to attention weights and to the output
    * `:key`     - PRNG key when dropout is active, else `nil`
    * `:return_weights` - also return the `{b, h, t, t}` attention
      distribution, for inspection. Off by default so the training path is
      untouched.
  """
  def self_attention(x, params, opts) do
    n_head = Keyword.fetch!(opts, :n_head)
    rope = Keyword.get(opts, :rope)
    rate = Keyword.get(opts, :dropout, 0.0)
    key = Keyword.get(opts, :key)
    return_weights = Keyword.get(opts, :return_weights, false)

    {b, t, d} = Nx.shape(x)
    head_dim = div(d, n_head)

    # {b, t, 3d} -> three {b, n_head, t, head_dim}
    qkv = Nx.dot(x, params["qkv"]["kernel"])
    q = split_heads(Nx.slice_along_axis(qkv, 0, d, axis: -1), n_head, head_dim)
    k = split_heads(Nx.slice_along_axis(qkv, d, d, axis: -1), n_head, head_dim)
    v = split_heads(Nx.slice_along_axis(qkv, 2 * d, d, axis: -1), n_head, head_dim)

    {q, k} =
      case rope do
        nil -> {q, k}
        tables -> {RoPE.apply_rotary(q, tables), RoPE.apply_rotary(k, tables)}
      end

    # {b, h, t, hd} x {b, h, t, hd} -> {b, h, t, t}
    scores =
      q
      |> Nx.dot([3], [0, 1], k, [3], [0, 1])
      |> Nx.multiply(1.0 / :math.sqrt(head_dim))

    # Causal mask: position i may only attend to positions <= i.
    mask =
      Nx.greater_equal(Nx.iota({t, t}, axis: 0), Nx.iota({t, t}, axis: 1))
      |> Nx.broadcast(Nx.shape(scores))

    scores = Nx.select(mask, scores, Nx.tensor(-1.0e9, type: Nx.type(scores)))

    {attn_key, out_key} = split_key(key)

    weights =
      scores
      |> Layers.softmax()
      |> maybe_dropout(attn_key, rate)

    # {b, h, t, t} x {b, h, t, hd} -> {b, h, t, hd}
    out = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    out =
      out
      # {b, h, t, hd} -> {b, t, h * hd}
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({b, t, d})
      |> Nx.dot(params["proj"]["kernel"])
      |> maybe_dropout(out_key, rate)

    if return_weights, do: {out, weights}, else: out
  end

  defp split_heads(x, n_head, head_dim) do
    {b, t, _d} = Nx.shape(x)

    x
    |> Nx.reshape({b, t, n_head, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp split_key(nil), do: {nil, nil}
  defp split_key(key), do: key |> Nx.Random.split() |> then(fn keys -> {keys[0], keys[1]} end)

  defp maybe_dropout(x, _key, rate) when rate == 0.0, do: x
  defp maybe_dropout(x, key, rate), do: Layers.dropout(x, key, rate)
end
