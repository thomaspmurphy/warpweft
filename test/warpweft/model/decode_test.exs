defmodule Warpweft.Model.DecodeTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Config, Model}
  alias Warpweft.Model.Decode

  @tiny %Config{
    vocab_size: 64,
    block_size: 16,
    n_layer: 2,
    n_head: 2,
    d_model: 32,
    dropout: 0.0
  }

  # Steps a whole sequence through the cache one token at a time, returning
  # the logits produced at each position.
  defp decode_all(params, ids, cfg) do
    {logits, _cache} =
      ids
      |> Enum.with_index()
      |> Enum.reduce({[], Decode.init_cache(cfg)}, fn {id, pos}, {acc, cache} ->
        {l, cache} =
          Decode.step(
            params,
            Nx.tensor([[id]], type: :s32),
            cache,
            Nx.tensor(pos, type: :s32),
            cfg
          )

        {[l | acc], cache}
      end)

    logits |> Enum.reverse() |> Nx.concatenate(axis: 0)
  end

  describe "step/5 agrees with the full forward pass" do
    for pos <- [:rope, :learned], mlp <- [:swiglu, :gelu], tie <- [true, false] do
      test "#{pos} / #{mlp} / tied=#{tie}" do
        cfg = %{@tiny | pos: unquote(pos), mlp: unquote(mlp), tie_embeddings: unquote(tie)}
        params = Model.init(cfg, Nx.Random.key(0))

        ids = [7, 3, 42, 11, 0, 63, 8, 19]

        full = Model.forward(params, Nx.tensor([ids], type: :s32), cfg)[[0]]
        cached = decode_all(params, ids, cfg)

        assert Nx.shape(cached) == Nx.shape(full)

        assert Nx.all_close(cached, full, atol: 1.0e-4) |> Nx.to_number() == 1,
               "max abs diff: #{full |> Nx.subtract(cached) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()}"
      end
    end
  end

  test "cache holds the keys and values of exactly the positions written" do
    cfg = @tiny
    params = Model.init(cfg, Nx.Random.key(0))
    cache = Decode.init_cache(cfg)

    {_l, cache} =
      Decode.step(params, Nx.tensor([[5]], type: :s32), cache, Nx.tensor(0, type: :s32), cfg)

    {_l, cache} =
      Decode.step(params, Nx.tensor([[9]], type: :s32), cache, Nx.tensor(1, type: :s32), cfg)

    k = cache["0"]["k"]
    assert Nx.shape(k) == {1, cfg.n_head, cfg.block_size, Config.head_dim(cfg)}

    # positions 0 and 1 written, everything past that still zero
    written = Nx.slice_along_axis(k, 0, 2, axis: 2)
    untouched = Nx.slice_along_axis(k, 2, cfg.block_size - 2, axis: 2)

    assert Nx.to_number(Nx.sum(Nx.abs(written))) > 0.0
    assert Nx.to_number(Nx.sum(Nx.abs(untouched))) == 0.0
  end

  test "init_cache has one entry per layer" do
    cache = Decode.init_cache(%{@tiny | n_layer: 5})
    assert Enum.sort(Map.keys(cache)) == ["0", "1", "2", "3", "4"]
  end
end
