defmodule Warpweft.ModelTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Config, Model}

  @tiny %Config{
    vocab_size: 64,
    block_size: 16,
    n_layer: 2,
    n_head: 2,
    d_model: 32,
    dropout: 0.1
  }

  defp variants do
    for pos <- [:rope, :learned],
        norm <- [:rms_norm, :layer_norm],
        mlp <- [:swiglu, :gelu],
        tie <- [true, false] do
      %{@tiny | pos: pos, norm: norm, mlp: mlp, tie_embeddings: tie}
    end
  end

  defp random_tokens(cfg, b, t, seed) do
    {tokens, _} =
      Nx.Random.randint(Nx.Random.key(seed), 0, cfg.vocab_size, shape: {b, t}, type: :s32)

    tokens
  end

  test "forward maps {b, t} -> {b, t, vocab} for all 16 variant combos" do
    for cfg <- variants() do
      params = Model.init(cfg, Nx.Random.key(0))
      tokens = random_tokens(cfg, 3, cfg.block_size, 1)
      logits = Model.forward(params, tokens, cfg)

      assert Nx.shape(logits) == {3, cfg.block_size, cfg.vocab_size},
             "bad shape for #{inspect({cfg.pos, cfg.norm, cfg.mlp, cfg.tie_embeddings})}"
    end
  end

  test "tied embeddings have no lm_head and fewer params" do
    tied = Model.init(%{@tiny | tie_embeddings: true}, Nx.Random.key(0))
    untied = Model.init(%{@tiny | tie_embeddings: false}, Nx.Random.key(0))

    refute Map.has_key?(tied, "lm_head")
    assert Model.param_count(untied) - Model.param_count(tied) == @tiny.vocab_size * @tiny.d_model
  end

  test "rope variant has no positional embedding table" do
    params = Model.init(%{@tiny | pos: :rope}, Nx.Random.key(0))
    refute Map.has_key?(params, "wpe")
  end

  describe "causal masking" do
    # The defining property of a decoder: logits at position t must not
    # change when tokens at positions > t change.
    for pos <- [:rope, :learned] do
      test "no future leakage with #{pos} positions" do
        cfg = %{@tiny | pos: unquote(pos)}
        params = Model.init(cfg, Nx.Random.key(0))

        t = cfg.block_size
        cut = div(t, 2)

        a = random_tokens(cfg, 2, t, 42)

        {suffix, _} =
          Nx.Random.randint(Nx.Random.key(99), 0, cfg.vocab_size, shape: {2, t - cut}, type: :s32)

        b = Nx.put_slice(a, [0, cut], suffix)

        # Sanity: the two inputs really differ after the cut.
        refute Nx.to_flat_list(a) == Nx.to_flat_list(b)

        logits_a = Model.forward(params, a, cfg)
        logits_b = Model.forward(params, b, cfg)

        assert Nx.all_close(
                 Nx.slice_along_axis(logits_a, 0, cut, axis: 1),
                 Nx.slice_along_axis(logits_b, 0, cut, axis: 1),
                 atol: 1.0e-5
               )
               |> Nx.to_number() == 1
      end
    end
  end

  test "dropout: same key reproduces, different keys differ, inference is deterministic" do
    cfg = %{@tiny | dropout: 0.2}
    params = Model.init(cfg, Nx.Random.key(0))
    tokens = random_tokens(cfg, 2, cfg.block_size, 7)

    k1 = Nx.Random.key(1)
    k2 = Nx.Random.key(2)

    same =
      Nx.all_close(
        Model.forward(params, tokens, cfg, key: k1),
        Model.forward(params, tokens, cfg, key: k1)
      )

    diff =
      Nx.all_close(
        Model.forward(params, tokens, cfg, key: k1),
        Model.forward(params, tokens, cfg, key: k2)
      )

    infer = Nx.all_close(Model.forward(params, tokens, cfg), Model.forward(params, tokens, cfg))

    assert Nx.to_number(same) == 1
    assert diff |> Nx.to_number() == 0
    assert Nx.to_number(infer) == 1
  end
end
