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

        {pad_junk, _} =
          Nx.Random.randint(Nx.Random.key(2), 0, 64, shape: {1, cfg.block_size - len}, type: :s32)

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

    # Temperature was previously untestable-by-accident: the argmax test
    # pinned top_k to 1 (so temperature could not change the outcome) and
    # the cached-vs-plain test cancels it on both sides. Deleting the
    # division entirely left the suite green.
    test "temperature controls how concentrated the sampling is" do
      params = params()
      buffer = Nx.broadcast(Nx.tensor(5, type: :s32), {1, @tiny.block_size})
      len = Nx.tensor(6, type: :s32)

      draws = fn temperature ->
        step = Generate.build_step(@tiny, temperature: temperature, top_k: nil)

        Enum.map_reduce(1..150, Nx.Random.key(0), fn _i, key ->
          {token, key} = step.(params, buffer, len, key)
          {Nx.to_number(token), key}
        end)
        |> elem(0)
      end

      cold = draws.(0.05)
      hot = draws.(5.0)

      most_common = cold |> Enum.frequencies() |> Enum.max_by(&elem(&1, 1)) |> elem(1)

      assert most_common > 140,
             "at temperature 0.05 sampling should collapse onto the argmax, got #{most_common}/150"

      assert length(Enum.uniq(hot)) > 10,
             "at temperature 5.0 sampling should spread out, got #{length(Enum.uniq(hot))} distinct tokens"
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

  describe "KV cache" do
    setup do
      corpus =
        String.duplicate("all the world is a stage and all the men and women merely players. ", 5)

      bpe = BPE.train(corpus, 280)
      cfg = %{@tiny | vocab_size: 280}
      %{bpe: bpe, cfg: cfg, params: Model.init(cfg, Nx.Random.key(0))}
    end

    test "cached and recomputing paths generate identical text", %{
      bpe: bpe,
      cfg: cfg,
      params: params
    } do
      for seed <- [1, 2, 99], temp <- [0.8, 1.5], top_k <- [nil, 5] do
        opts = [max_new_tokens: 8, seed: seed, temperature: temp, top_k: top_k]

        cached = Generate.generate(params, bpe, cfg, "all the", opts ++ [cache: true])
        plain = Generate.generate(params, bpe, cfg, "all the", opts ++ [cache: false])

        assert cached == plain,
               "diverged at seed=#{seed} temp=#{temp} top_k=#{inspect(top_k)}"
      end
    end

    test "falls back to the recomputing path when the context would overflow", %{cfg: cfg} do
      assert Generate.cacheable?(cfg, 4, 8)
      refute Generate.cacheable?(cfg, 4, cfg.block_size)
    end

    test "cached generation stops at the block limit rather than corrupting the cache", %{
      bpe: bpe,
      cfg: cfg,
      params: params
    } do
      # This must genuinely take the cached path, so prompt + n has to fit
      # in block_size while still running pos up to the limit. An earlier
      # version asked for block_size * 2 tokens, which made `fits` false
      # and quietly exercised the recomputing path instead.
      prompt = "all"
      prompt_len = length(BPE.encode(bpe, prompt))
      n = cfg.block_size - prompt_len

      assert Generate.cacheable?(cfg, prompt_len, n), "this test must exercise the cached path"

      text = Generate.generate(params, bpe, cfg, prompt, max_new_tokens: n, cache: true, seed: 1)
      generated = String.replace_prefix(text, prompt, "")

      # Every requested token is produced, and the run halts at the limit
      # rather than writing past the end of the cache.
      assert length(BPE.encode(bpe, generated)) == n

      # And it still matches the recomputing path exactly.
      plain =
        Generate.generate(params, bpe, cfg, prompt, max_new_tokens: n, cache: false, seed: 1)

      assert text == plain
    end
  end

  describe "streaming via :on_token" do
    # Byte-level tokens can end mid-codepoint, so a naive streamer would
    # print replacement characters. Train on text full of multi-byte
    # characters so the vocabulary is riddled with partial sequences.
    setup do
      corpus = String.duplicate("le café était très naïf — 日本語 🎉 ", 20)
      bpe = BPE.train(corpus, 300)
      # Highly repetitive corpora run out of pairs before reaching the
      # requested size, so take the vocabulary the tokenizer actually has.
      cfg = %{@tiny | vocab_size: BPE.vocab_size(bpe)}
      %{bpe: bpe, cfg: cfg, params: Model.init(cfg, Nx.Random.key(0))}
    end

    test "every emitted chunk is valid UTF-8", %{bpe: bpe, cfg: cfg, params: params} do
      for seed <- 1..10 do
        {:ok, agent} = Agent.start_link(fn -> [] end)

        Generate.generate(params, bpe, cfg, "le",
          max_new_tokens: 20,
          seed: seed,
          on_token: fn chunk -> Agent.update(agent, &[chunk | &1]) end
        )

        chunks = agent |> Agent.get(&Enum.reverse/1)
        Agent.stop(agent)

        # This also proves the hold-back works: had a multi-byte character
        # been split across two chunks, the first would end in a lone lead
        # byte and fail validity here.
        for chunk <- chunks do
          assert String.valid?(chunk), "invalid UTF-8 chunk #{inspect(chunk)} at seed #{seed}"
        end
      end
    end

    # The base alphabet is all 256 bytes whatever the corpus, so a randomly
    # initialized model emits malformed sequences constantly and the stream
    # deliberately substitutes U+FFFD for them. Byte-exact equality with
    # the return value therefore cannot hold in general. What must hold is
    # that streaming loses, duplicates and reorders nothing — so compare
    # against the return value put through the same substitution. Keeping
    # that policy written out here means changing it in the implementation
    # will fail this test, which is the point.
    defp scrub(bin) do
      case :unicode.characters_to_binary(bin) do
        valid when is_binary(valid) -> valid
        {:incomplete, valid, _rest} -> valid <> "�"
        {:error, valid, <<_bad, rest::binary>>} -> valid <> "�" <> scrub(rest)
      end
    end

    for cache <- [true, false] do
      test "streaming loses nothing (cache=#{cache})", %{bpe: bpe, cfg: cfg, params: params} do
        for seed <- 1..5 do
          {:ok, agent} = Agent.start_link(fn -> [] end)

          text =
            Generate.generate(params, bpe, cfg, "le café",
              max_new_tokens: 30,
              seed: seed,
              cache: unquote(cache),
              on_token: fn chunk -> Agent.update(agent, &[chunk | &1]) end
            )

          streamed = agent |> Agent.get(&Enum.reverse/1) |> IO.iodata_to_binary()
          Agent.stop(agent)

          assert "le café" <> streamed == scrub(text), "mismatch at seed #{seed}"
        end
      end
    end
  end

  test "end-to-end generate round-trips through the tokenizer" do
    corpus =
      String.duplicate("all the world is a stage and all the men and women merely players. ", 5)

    bpe = BPE.train(corpus, 280)
    cfg = %{@tiny | vocab_size: 280}
    params = Model.init(cfg, Nx.Random.key(0))

    text = Generate.generate(params, bpe, cfg, "all the", max_new_tokens: 8, seed: 1)

    assert String.starts_with?(text, "all the")
    assert String.length(text) > String.length("all the")
  end
end
