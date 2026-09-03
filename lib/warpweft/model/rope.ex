defmodule Warpweft.Model.RoPE do
  @moduledoc """
  Rotary positional embeddings (Su et al., 2021), GPT-NeoX "half-split" style.

  Instead of adding a learned position vector to the token embedding, RoPE
  rotates each (query, key) head vector by an angle proportional to its
  position. Dot products between rotated q and k then depend only on the
  *relative* distance between positions, which generalizes better and adds
  zero parameters.
  """

  @theta 10_000.0

  @doc """
  cos/sin tables of shape `{seq_len, head_dim / 2}`.

  Computed from iota at trace time, so they fold into the compiled program
  as constants for a fixed sequence length.
  """
  def tables(seq_len, head_dim) do
    half = div(head_dim, 2)

    inv_freq =
      Nx.pow(@theta, Nx.iota({half}, type: :f32) |> Nx.divide(half) |> Nx.negate())

    angles =
      {seq_len}
      |> Nx.iota(type: :f32)
      |> Nx.new_axis(1)
      |> Nx.multiply(Nx.new_axis(inv_freq, 0))

    {Nx.cos(angles), Nx.sin(angles)}
  end

  @doc """
  Applies the rotation to a `{batch, heads, seq, head_dim}` tensor.

  The head dim is treated as two halves `[x1, x2]`; each pair
  `(x1[i], x2[i])` is a 2-D point rotated by the angle for its position:

      rotated = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
  """
  def apply_rotary(x, {cos, sin}) do
    {_b, _h, _t, head_dim} = Nx.shape(x)
    half = div(head_dim, 2)

    x1 = Nx.slice_along_axis(x, 0, half, axis: -1)
    x2 = Nx.slice_along_axis(x, half, half, axis: -1)

    # {t, half} -> {1, 1, t, half} for broadcasting
    cos = cos |> Nx.new_axis(0) |> Nx.new_axis(0)
    sin = sin |> Nx.new_axis(0) |> Nx.new_axis(0)

    r1 = x1 |> Nx.multiply(cos) |> Nx.subtract(Nx.multiply(x2, sin))
    r2 = x2 |> Nx.multiply(cos) |> Nx.add(Nx.multiply(x1, sin))

    Nx.concatenate([r1, r2], axis: -1)
  end
end
