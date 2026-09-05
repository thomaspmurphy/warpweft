defmodule Warpweft.Model.RoPETest do
  use ExUnit.Case, async: true

  alias Warpweft.Model.RoPE

  # The entire justification for rotary embeddings is that after rotating
  # queries and keys by their absolute positions, their dot product depends
  # only on the *relative* distance between those positions. Nothing else
  # in the suite checks that: the decode tests would still pass if the
  # rotation were wrong but consistently wrong across both code paths.

  defp rotate_at(vector, pos, seq_len, head_dim) do
    {cos, sin} = RoPE.tables(seq_len, head_dim)
    idx = Nx.tensor([pos])
    RoPE.apply_rotary(vector, {Nx.take(cos, idx), Nx.take(sin, idx)})
  end

  defp score(q, k, i, j, seq_len, head_dim) do
    qi = rotate_at(q, i, seq_len, head_dim)
    kj = rotate_at(k, j, seq_len, head_dim)

    qi |> Nx.multiply(kj) |> Nx.sum() |> Nx.to_number()
  end

  test "the score depends only on the distance between positions" do
    head_dim = 16
    seq_len = 64
    shape = {1, 1, 1, head_dim}

    {q, key} = Nx.Random.normal(Nx.Random.key(0), shape: shape)
    {k, _key} = Nx.Random.normal(key, shape: shape)

    for distance <- [0, 1, 5, 17] do
      scores =
        for i <- [distance, distance + 3, distance + 11, distance + 40] do
          score(q, k, i, i - distance, seq_len, head_dim)
        end

      [reference | rest] = scores

      for s <- rest do
        assert_in_delta s, reference, 1.0e-4
      end
    end
  end

  test "different distances give different scores" do
    head_dim = 16
    shape = {1, 1, 1, head_dim}
    {q, key} = Nx.Random.normal(Nx.Random.key(1), shape: shape)
    {k, _key} = Nx.Random.normal(key, shape: shape)

    at_0 = score(q, k, 10, 10, 64, head_dim)
    at_7 = score(q, k, 17, 10, 64, head_dim)

    refute_in_delta at_0, at_7, 1.0e-3
  end

  test "rotation preserves vector norm" do
    shape = {1, 2, 4, 16}
    {q, _key} = Nx.Random.normal(Nx.Random.key(2), shape: shape)

    {cos, sin} = RoPE.tables(4, 16)
    rotated = RoPE.apply_rotary(q, {cos, sin})

    before_norm = q |> Nx.pow(2) |> Nx.sum(axes: [-1])
    after_norm = rotated |> Nx.pow(2) |> Nx.sum(axes: [-1])

    assert Nx.all_close(before_norm, after_norm, atol: 1.0e-4) |> Nx.to_number() == 1
  end

  test "position 0 is the identity rotation" do
    shape = {1, 1, 1, 8}
    {q, _key} = Nx.Random.normal(Nx.Random.key(3), shape: shape)

    assert Nx.all_close(rotate_at(q, 0, 8, 8), q, atol: 1.0e-6) |> Nx.to_number() == 1
  end
end
