defmodule Warpweft.Model.Decode do
  @moduledoc """
  Single-position forward pass for autoregressive decoding, using a
  key/value cache.

  `Warpweft.Model.forward/4` processes a whole sequence at once, which is
  what you want for training: every position's loss is computed in one
  pass. For generation it is wasteful: to produce token `n + 1` it
  recomputes everything about tokens `0..n`, which cannot have changed.

  `step/5` instead runs one token through the network, reading the
  previous positions' keys and values from a cache and appending its own.
  The arithmetic per token drops from O(context^2) to O(context).

  ## Context limit

  The cache is a fixed `block_size`-wide buffer, so decoding this way
  works up to `block_size` positions total. It cannot slide the window
  the way a full forward pass can: cached keys were rotated (RoPE) or
  offset (learned positions) at their original absolute positions, so
  shifting them left would silently corrupt those encodings.
  `Warpweft.Generate` therefore uses this path while the context fits and
  falls back to the recomputing path beyond it.
  """

  alias Warpweft.Config
  alias Warpweft.Model.{Attention, Layers, RoPE}

  @doc "Zeroed cache: per layer, `k` and `v` of `{1, n_head, block_size, head_dim}`."
  def init_cache(%Config{} = cfg) do
    shape = {1, cfg.n_head, cfg.block_size, Config.head_dim(cfg)}
    empty = %{"k" => Nx.broadcast(0.0, shape), "v" => Nx.broadcast(0.0, shape)}

    for i <- 0..(cfg.n_layer - 1), into: %{}, do: {Integer.to_string(i), empty}
  end

  @doc """
  Runs one token at absolute position `pos`, returning
  `{logits, updated_cache}` where logits are `{1, vocab_size}`, the
  prediction for position `pos + 1`.

  `token` is `{1, 1}`; `pos` is a scalar tensor so the compiled program is
  reused for every position.
  """
  def step(params, token, cache, pos, %Config{} = cfg) do
    x = Nx.take(params["wte"]["kernel"], token)

    x =
      case cfg.pos do
        :learned ->
          position = params["wpe"]["kernel"] |> Nx.take(Nx.reshape(pos, {1})) |> Nx.reshape({1, 1, cfg.d_model})
          Nx.add(x, position)

        :rope ->
          x
      end

    rope =
      if cfg.pos == :rope do
        {cos, sin} = RoPE.tables(cfg.block_size, Config.head_dim(cfg))
        idx = Nx.reshape(pos, {1})
        {Nx.take(cos, idx), Nx.take(sin, idx)}
      end

    {x, cache} =
      Enum.reduce(0..(cfg.n_layer - 1), {x, cache}, fn i, {x, cache} ->
        name = Integer.to_string(i)
        block = params["blocks"][name]

        {attn_out, layer_cache} =
          x
          |> Layers.norm(block["norm1"], cfg.norm)
          |> Attention.cached_attention(block["attn"], cache[name], pos,
            n_head: cfg.n_head,
            rope: rope
          )

        x = Nx.add(x, attn_out)

        mlp_out =
          x
          |> Layers.norm(block["norm2"], cfg.norm)
          |> Layers.mlp(block["mlp"], cfg.mlp, nil, 0.0)

        {Nx.add(x, mlp_out), Map.put(cache, name, layer_cache)}
      end)

    x = Layers.norm(x, params["final_norm"], cfg.norm)

    logits =
      if cfg.tie_embeddings do
        Nx.dot(x, Nx.transpose(params["wte"]["kernel"]))
      else
        Nx.dot(x, params["lm_head"]["kernel"])
      end

    {Nx.reshape(logits, {1, cfg.vocab_size}), cache}
  end
end
