defmodule Warpweft.GenerateTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Config, Generate, Model}
  alias Warpweft.Tokenizer.BPE

  @tiny %Config{
    vocab_size: 64,
    block_size: 16,
    n_layer: 2,
    n_head: 2,
    d_model: 32,
    dropout: 0.0
  }

  defp params, do: Model.init(@tiny, Nx.Random.key(0))

  describe "padding safety (what the fixed-shape buffer relies on)" do
    for pos <- [:rope, :learned] do
      test "logits at len-1 ignore everything after len (#{pos})" do
        cfg = %{@tiny | pos: unquote(pos)}
        params = Model.init(cfg, Nx.Random.key(0))

        len = 5
        {prefix, _} = Nx.Random.randint(Nx.Random.key(1), 0, 64, shape: {1, len}, type: :s32)

        pad_zeros = Nx.broadcast(Nx.tensor(0, type: :s32), {1, cfg.block_size - len})
        {pad_junk, _} = Nx.Random.randint(Nx.Random.key(2), 0, 64, shape: {1, cfg.block_size - len}, type: :s32)

        a = Nx.concatenate([prefix, pad_zeros], axis: 1)
        b = Nx.concatenate([prefix, pad_junk], axis: 1)

        la = Model.forward(params, a, cfg)[[0, len - 1]]
        lb = Model.forward(params, b, cfg)[[0, len - 1]]

        assert Nx.all_close(la, lb, atol: 1.0e-5) |> Nx.to_number() == 1
      end
    end
  end

  describe "sampling step" do
    test "temperature -> 0 with top_k 1 equals argmax" do
      step = Generate.build_step(@tiny, temperature: 0.01, top_k: 1)
      params = params()

      buffer = Nx.broadcast(Nx.tensor(1, type: :s32), {1, @tiny.block_size})
      len = Nx.tensor(4, type: :s32)

      logits = Model.forward(params, buffer, @tiny)
      expected = logits[[0, 3]] |> Nx.argmax() |> Nx.to_number()

      for seed <- 1..5 do
        {token, _key} = step.(params, buffer, len, Nx.Random.key(seed))
        assert Nx.to_number(token) == expected
      end
    end

    test "samples never escape the top-k support" do
      k = 3
      step = Generate.build_step(@tiny, temperature: 1.5, top_k: k)
      params = params()

      buffer = Nx.broadcast(Nx.tensor(2, type: :s32), {1, @tiny.block_size})
      len = Nx.tensor(8, type: :s32)

      {_, top_idx} = Model.forward(params, buffer, @tiny)[[0, 7]] |> Nx.top_k(k: k)
      allowed = top_idx |> Nx.to_flat_list() |> MapSet.new()

      key0 = Nx.Random.key(0)

      Enum.reduce(1..200, key0, fn _, key ->
        {token, key} = step.(params, buffer, len, key)
        assert MapSet.member?(allowed, Nx.to_number(token))
        key
      end)
    end

    test "different seeds produce different continuations at high temperature" do
      step = Generate.build_step(@tiny, temperature: 1.5, top_k: nil)
      params = params()
      buffer = Nx.broadcast(Nx.tensor(3, type: :s32), {1, @tiny.block_size})
      len = Nx.tensor(4, type: :s32)

      tokens =
        for seed <- 1..20 do
          {token, _} = step.(params, buffer, len, Nx.Random.key(seed))
          Nx.to_number(token)
        end

      assert tokens |> Enum.uniq() |> length() > 1
    end
  end

  test "end-to-end generate round-trips through the tokenizer" do
    corpus = String.duplicate("all the world is a stage and all the men and women merely players. ", 5)
    bpe = BPE.train(corpus, 280)
    cfg = %{@tiny | vocab_size: 280}
    params = Model.init(cfg, Nx.Random.key(0))

    text = Generate.generate(params, bpe, cfg, "all the", max_new_tokens: 8, seed: 1)

    assert String.starts_with?(text, "all the")
    assert String.length(text) > String.length("all the")
  end
end
