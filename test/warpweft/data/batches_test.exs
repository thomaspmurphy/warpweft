defmodule Warpweft.Data.BatchesTest do
  use ExUnit.Case, async: true

  alias Warpweft.Data.Batches

  defp data, do: Nx.iota({1000}, type: :s32)

  test "sample shapes and target shift" do
    {x, y, _key} = Batches.sample(data(), Nx.Random.key(0), 8, 32)

    assert Nx.shape(x) == {8, 32}
    assert Nx.shape(y) == {8, 32}

    # y is x shifted one position left
    assert Nx.to_flat_list(Nx.slice_along_axis(x, 1, 31, axis: 1)) ==
             Nx.to_flat_list(Nx.slice_along_axis(y, 0, 31, axis: 1))

    # data is iota, so every target is input + 1
    assert Nx.all(Nx.equal(y, Nx.add(x, 1))) |> Nx.to_number() == 1
  end

  test "all sampled indices stay in range" do
    {x, y, _key} = Batches.sample(data(), Nx.Random.key(123), 64, 100)

    assert Nx.to_number(Nx.reduce_min(x)) >= 0
    assert Nx.to_number(Nx.reduce_max(y)) <= 999
  end

  test "different keys give different batches; same key reproduces" do
    {x1, _, _} = Batches.sample(data(), Nx.Random.key(1), 8, 32)
    {x2, _, _} = Batches.sample(data(), Nx.Random.key(2), 8, 32)
    {x3, _, _} = Batches.sample(data(), Nx.Random.key(1), 8, 32)

    refute Nx.to_flat_list(x1) == Nx.to_flat_list(x2)
    assert Nx.to_flat_list(x1) == Nx.to_flat_list(x3)
  end

  test "stream yields distinct batches and fresh dropout keys" do
    [{x1, _, k1}, {x2, _, k2}] = data() |> Batches.stream(0, 4, 16) |> Enum.take(2)

    refute Nx.to_flat_list(x1) == Nx.to_flat_list(x2)
    refute Nx.to_flat_list(k1) == Nx.to_flat_list(k2)
  end
end
