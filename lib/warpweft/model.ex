defmodule Warpweft.Model do
  @moduledoc """
  The decoder-only transformer: parameter initialisation and forward pass.

  Parameters live in a visible nested map (no framework graph):

      %{
        "wte" => %{"kernel" => {vocab, d}},          # token embedding
        "wpe" => %{"kernel" => {block, d}},          # only when pos: :learned
        "blocks" => %{"0" => block, "1" => ...},
        "final_norm" => %{"gamma" => {d}, ...},
        "lm_head" => %{"kernel" => {d, vocab}}       # only when not tied
      }

  The forward pass is plain Elixir over Nx tensors; wrapped in
  `Nx.Defn.jit/2` (see `Warpweft.Train` / `Warpweft.Generate`) it traces
  into a single fused XLA program. All architecture variants are decided
  at trace time from the config, so there is no runtime branching.

  Initialisation is normal(0, 0.02) everywhere, with the residual output
  projections scaled down by 1/sqrt(2 * n_layer) so the residual stream
  variance stays stable with depth.
  """

  alias Warpweft.Config
  alias Warpweft.Model.{Attention, Layers, RoPE}

  @init_std 0.02

  # -- init ---------------------------------------------------------------------

  def init(%Config{} = cfg, key) do
    Config.validate!(cfg)
    d = cfg.d_model
    resid_std = @init_std / :math.sqrt(2 * cfg.n_layer)

    {wte, key} = normal(key, {cfg.vocab_size, d}, @init_std)

    {wpe, key} =
      case cfg.pos do
        :learned -> normal(key, {cfg.block_size, d}, @init_std)
        :rope -> {nil, key}
      end

    {blocks, key} =
      Enum.reduce(0..(cfg.n_layer - 1), {%{}, key}, fn i, {blocks, key} ->
        {block, key} = init_block(cfg, key, resid_std)
        {Map.put(blocks, Integer.to_string(i), block), key}
      end)

    {lm_head, _key} =
      if cfg.tie_embeddings, do: {nil, key}, else: normal(key, {d, cfg.vocab_size}, @init_std)

    %{
      "wte" => %{"kernel" => wte},
      "blocks" => blocks,
      "final_norm" => init_norm(cfg.norm, d)
    }
    |> put_unless_nil("wpe", wpe && %{"kernel" => wpe})
    |> put_unless_nil("lm_head", lm_head && %{"kernel" => lm_head})
  end

  defp init_block(%Config{} = cfg, key, resid_std) do
    d = cfg.d_model

    {qkv, key} = normal(key, {d, 3 * d}, @init_std)
    {attn_proj, key} = normal(key, {d, d}, resid_std)

    {mlp, key} =
      case cfg.mlp do
        :gelu ->
          {fc, key} = normal(key, {d, 4 * d}, @init_std)
          {proj, key} = normal(key, {4 * d, d}, resid_std)
          {%{"fc" => %{"kernel" => fc}, "proj" => %{"kernel" => proj}}, key}

        :swiglu ->
          h = Config.swiglu_hidden(cfg)
          {w1, key} = normal(key, {d, h}, @init_std)
          {w3, key} = normal(key, {d, h}, @init_std)
          {w2, key} = normal(key, {h, d}, resid_std)

          {%{"w1" => %{"kernel" => w1}, "w2" => %{"kernel" => w2}, "w3" => %{"kernel" => w3}},
           key}
      end

    block = %{
      "norm1" => init_norm(cfg.norm, d),
      "norm2" => init_norm(cfg.norm, d),
      "attn" => %{"qkv" => %{"kernel" => qkv}, "proj" => %{"kernel" => attn_proj}},
      "mlp" => mlp
    }

    {block, key}
  end

  defp init_norm(:rms_norm, d), do: %{"gamma" => Nx.broadcast(1.0, {d})}

  defp init_norm(:layer_norm, d),
    do: %{"gamma" => Nx.broadcast(1.0, {d}), "beta" => Nx.broadcast(0.0, {d})}

  defp normal(key, shape, std) do
    {t, key} = Nx.Random.normal(key, 0.0, std, shape: shape, type: :f32)
    {t, key}
  end

  defp put_unless_nil(map, _k, nil), do: map
  defp put_unless_nil(map, k, v), do: Map.put(map, k, v)

  # -- forward ------------------------------------------------------------------

  @doc """
  Runs tokens `{batch, seq}` through the model, returning logits
  `{batch, seq, vocab}`.

  Pass `key: prng_key` to enable dropout (training); omit it for the
  inference path, where every dropout is traced away entirely.

  Pass `collect_attention: true` to get `{logits, attentions}` instead of
  bare logits, where `attentions` is a list of `{b, h, t, t}` tensors, one
  per layer, in layer order. Used by `mix wf.attention`.
  """
  def forward(params, tokens, %Config{} = cfg, opts \\ []) do
    key = Keyword.get(opts, :key)
    collect = Keyword.get(opts, :collect_attention, false)
    rate = if key, do: cfg.dropout, else: 0.0
    {_b, t} = Nx.shape(tokens)

    x = Nx.take(params["wte"]["kernel"], tokens)

    x =
      case cfg.pos do
        :learned -> Nx.add(x, Nx.take(params["wpe"]["kernel"], Nx.iota({t})))
        :rope -> x
      end

    rope = if cfg.pos == :rope, do: RoPE.tables(t, Config.head_dim(cfg))

    {emb_key, key} = split_key(key)
    x = maybe_dropout(x, emb_key, rate)

    {x, _key, attentions} =
      Enum.reduce(0..(cfg.n_layer - 1), {x, key, []}, fn i, {x, key, attns} ->
        {attn_key, key} = split_key(key)
        {mlp_key, key} = split_key(key)
        block = params["blocks"][Integer.to_string(i)]

        attn_in = Layers.norm(x, block["norm1"], cfg.norm)

        attn_opts = [n_head: cfg.n_head, rope: rope, dropout: rate, key: attn_key]

        {attn_out, attns} =
          if collect do
            {out, weights} =
              Attention.self_attention(
                attn_in,
                block["attn"],
                [return_weights: true] ++ attn_opts
              )

            {out, [weights | attns]}
          else
            {Attention.self_attention(attn_in, block["attn"], attn_opts), attns}
          end

        x = Nx.add(x, attn_out)

        mlp_in = Layers.norm(x, block["norm2"], cfg.norm)
        x = Nx.add(x, Layers.mlp(mlp_in, block["mlp"], cfg.mlp, mlp_key, rate))

        {x, key, attns}
      end)

    x = Layers.norm(x, params["final_norm"], cfg.norm)

    logits =
      if cfg.tie_embeddings do
        Nx.dot(x, Nx.transpose(params["wte"]["kernel"]))
      else
        Nx.dot(x, params["lm_head"]["kernel"])
      end

    if collect, do: {logits, Enum.reverse(attentions)}, else: logits
  end

  @doc "Number of parameters in a params map."
  def param_count(params) do
    params
    |> flatten_params()
    |> Enum.map(fn {_path, t} -> Nx.size(t) end)
    |> Enum.sum()
  end

  defp flatten_params(map, path \\ []) do
    Enum.flat_map(map, fn {k, v} ->
      case v do
        %Nx.Tensor{} -> [{Enum.reverse([k | path]), v}]
        %{} -> flatten_params(v, [k | path])
      end
    end)
  end

  defp split_key(nil), do: {nil, nil}
  defp split_key(key), do: key |> Nx.Random.split() |> then(fn keys -> {keys[0], keys[1]} end)

  defp maybe_dropout(x, _key, rate) when rate == 0.0, do: x
  defp maybe_dropout(x, key, rate), do: Layers.dropout(x, key, rate)
end
