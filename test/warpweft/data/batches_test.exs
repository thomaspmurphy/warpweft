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

  # `Nx.take` clamps out-of-range gather indices rather than raising, so
  # asserting min/max bounds cannot fail no matter how wrong the offsets
  # are. What clamping actually does is repeat the final token, which
  # shows up as a zero step between consecutive positions.
  test "every sampled window is strictly consecutive, even hard against the end" do
    small = Nx.iota({40}, type: :s32)

    for seed <- 1..40 do
      {x, _y, _key} = Batches.sample(small, Nx.Random.key(seed), 8, 32)

      steps =
        Nx.subtract(
          Nx.slice_along_axis(x, 1, 31, axis: 1),
          Nx.slice_along_axis(x, 0, 31, axis: 1)
        )

      assert Nx.to_number(Nx.reduce_min(steps)) == 1,
             "window is not consecutive at seed #{seed}: offsets ran past the corpus and were clamped"
    end
  end

  test "the final corpus token is reachable as a target" do
    data = Nx.iota({200}, type: :s32)

    max_target =
      Enum.reduce(1..200, 0, fn seed, acc ->
        {_x, y, _key} = Batches.sample(data, Nx.Random.key(seed), 8, 8)
        max(acc, Nx.to_number(Nx.reduce_max(y)))
      end)

    assert max_target == 199, "off-by-one: the last token is never predicted"
  end

  test "a corpus too short for the block size raises rather than sampling garbage" do
    assert_raise ArgumentError, ~r/too short for block_size/, fn ->
      Batches.sample(Nx.iota({100}, type: :s32), Nx.Random.key(1), 4, 128)
    end
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
