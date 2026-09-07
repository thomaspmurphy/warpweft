defmodule Warpweft.IntrospectTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Config, Introspect, Model}

  @tiny %Config{
    vocab_size: 64,
    block_size: 16,
    n_layer: 2,
    n_head: 2,
    d_model: 32,
    dropout: 0.0
  }

  defp weights_for(cfg) do
    params = Model.init(cfg, Nx.Random.key(0))

    {tokens, _} =
      Nx.Random.randint(Nx.Random.key(1), 0, cfg.vocab_size, shape: {1, 8}, type: :s32)

    {logits, weights} = Model.forward(params, tokens, cfg, collect_attention: true)
    {logits, weights}
  end

  test "collect_attention returns one {b, h, t, t} tensor per layer" do
    {logits, weights} = weights_for(@tiny)

    assert Nx.shape(logits) == {1, 8, @tiny.vocab_size}
    assert length(weights) == @tiny.n_layer
    assert Enum.all?(weights, &(Nx.shape(&1) == {1, @tiny.n_head, 8, 8}))
  end

  test "collecting attention does not change the logits" do
    params = Model.init(@tiny, Nx.Random.key(0))
    {tokens, _} = Nx.Random.randint(Nx.Random.key(1), 0, 64, shape: {1, 8}, type: :s32)

    plain = Model.forward(params, tokens, @tiny)
    {collected, _weights} = Model.forward(params, tokens, @tiny, collect_attention: true)

    assert Nx.all_close(plain, collected) |> Nx.to_number() == 1
  end

  test "weights are probability distributions over visible positions" do
    {_logits, weights} = weights_for(@tiny)

    for w <- weights do
      # every query row sums to 1
      sums = Nx.sum(w, axes: [-1])

      assert Nx.all_close(sums, Nx.broadcast(1.0, Nx.shape(sums)), atol: 1.0e-5) |> Nx.to_number() ==
               1

      # nothing is negative
      assert Nx.to_number(Nx.reduce_min(w)) >= 0.0

      # strictly causal: no mass above the diagonal
      upper = Nx.less(Nx.iota({8, 8}, axis: 0), Nx.iota({8, 8}, axis: 1))
      masked = Nx.select(Nx.broadcast(upper, Nx.shape(w)), w, 0.0)
      assert Nx.to_number(Nx.sum(masked)) < 1.0e-6
    end
  end

  describe "head_stats/1" do
    test "identifies a pure previous-token head" do
      # weight 1 on position i-1 (row 0 attends to itself)
      t = 6
      offset = Nx.subtract(Nx.iota({t, t}, axis: 0), Nx.iota({t, t}, axis: 1))
      prev = Nx.equal(offset, 1) |> Nx.as_type(:f32)
      w = Nx.put_slice(prev, [0, 0], Nx.tensor([[1.0]]))

      stats = Introspect.head_stats(w)

      assert_in_delta stats.prev, 1.0, 1.0e-6
      assert_in_delta stats.distance, (t - 1) / t, 1.0e-6
      assert_in_delta stats.entropy, 0.0, 1.0e-6
      assert Introspect.classify(stats) == "previous-token"
    end

    test "identifies a pure sink head" do
      t = 6
      w = Nx.equal(Nx.iota({t, t}, axis: 1), 0) |> Nx.as_type(:f32)
      stats = Introspect.head_stats(w)

      assert_in_delta stats.sink, 1.0, 1.0e-6
      assert Introspect.classify(stats) == "sink (position 0)"
    end

    test "a uniform causal head has entropy near 1 and is called diffuse" do
      t = 8
      rows = Nx.iota({t, t}, axis: 0)
      cols = Nx.iota({t, t}, axis: 1)
      causal = Nx.greater_equal(rows, cols) |> Nx.as_type(:f32)
      w = Nx.divide(causal, Nx.sum(causal, axes: [1], keep_axes: true))

      stats = Introspect.head_stats(w)

      assert_in_delta stats.entropy, 1.0, 0.01
      assert Introspect.classify(stats) == "diffuse"
    end
  end
end
