defmodule Warpweft.Data.Batches do
  @moduledoc """
  Random training batches, sampled entirely on device.

  One `Nx.take` gathers a whole `{batch, block + 1}` window matrix from the
  flat corpus tensor: random offsets broadcast against an iota give every
  window's absolute indices, and slicing off the first/last column yields
  the inputs and next-token targets. Fixed shapes mean XLA compiles this
  exactly once, and nothing crosses back to the host per batch.
  """

  @doc """
  Samples `{x, y, new_key}` where `x`/`y` are `{batch, block}` and `y` is
  `x` shifted one token left. Traceable; JIT once and reuse.
  """
  def sample(data, key, batch, block) do
    n = Nx.size(data)

    {offsets, key} = Nx.Random.randint(key, 0, n - block - 1, shape: {batch}, type: :s32)

    idx =
      offsets
      |> Nx.new_axis(1)
      |> Nx.add(Nx.iota({1, block + 1}))

    windows = Nx.take(data, idx)

    x = Nx.slice_along_axis(windows, 0, block, axis: 1)
    y = Nx.slice_along_axis(windows, 1, block, axis: 1)
    {x, y, key}
  end

  @doc """
  Infinite stream of `{x, y, dropout_key}` batches from a jitted sampler.

  Each element also carries a fresh PRNG key for that step's dropout, so
  no two steps ever reuse a dropout mask.
  """
  def stream(data, seed, batch, block) do
    sampler = Nx.Defn.jit(&sample(&1, &2, batch, block))
    splitter = Nx.Defn.jit(&Nx.Random.split/1)

    Stream.unfold(Nx.Random.key(seed), fn key ->
      keys = splitter.(key)
      {x, y, next_key} = sampler.(data, keys[0])
      {{x, y, keys[1]}, next_key}
    end)
  end
end
