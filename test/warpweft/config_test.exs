defmodule Warpweft.ConfigTest do
  use ExUnit.Case, async: true

  alias Warpweft.Config

  @moduletag :tmp_dir

  test "every preset round-trips through save and load", %{tmp_dir: dir} do
    for name <- Config.presets() do
      config = Config.preset(name)
      Config.save(config, dir)
      assert Config.load(dir) == config, "preset #{name} did not survive the round trip"
    end
  end

  test "a fully non-default config round-trips", %{tmp_dir: dir} do
    config = %Config{
      corpus: "custom",
      vocab_size: 512,
      block_size: 64,
      n_layer: 3,
      n_head: 2,
      d_model: 128,
      dropout: 0.25,
      pos: :learned,
      norm: :layer_norm,
      mlp: :gelu,
      tie_embeddings: false,
      seed: 7
    }

    Config.save(config, dir)
    assert Config.load(dir) == config
  end

  # load/1 converts only the fields listed in atom_fields back from
  # strings. Adding a fourth atom-valued field and forgetting to list it
  # would make load/1 return a string that survives until the model tries
  # to dispatch on it, far from the cause.
  test "atom_fields lists exactly the atom-valued struct fields" do
    actual =
      %Config{}
      |> Map.from_struct()
      |> Enum.filter(fn {_k, v} -> is_atom(v) and not is_boolean(v) and not is_nil(v) end)
      |> Enum.map(fn {k, _v} -> k end)
      |> Enum.sort()

    assert actual == Enum.sort(Config.atom_fields()),
           "Config.atom_fields/0 is out of date; load/1 will return strings for the missing fields"
  end

  describe "load/1 error messages" do
    test "names an unknown field", %{tmp_dir: dir} do
      File.write!(Path.join(dir, "config.json"), ~s({"corpus":"x","not_a_field":1}))

      assert_raise ArgumentError, ~r/unknown field "not_a_field"/, fn -> Config.load(dir) end
    end

    test "names a bad variant value and lists the valid ones", %{tmp_dir: dir} do
      File.write!(Path.join(dir, "config.json"), ~s({"pos":"telepathy"}))

      assert_raise ArgumentError, ~r/pos: "telepathy", expected one of rope, learned/, fn ->
        Config.load(dir)
      end
    end
  end

  describe "validate!/1" do
    test "rejects d_model not divisible by n_head" do
      assert_raise ArgumentError, ~r/not divisible/, fn ->
        Config.validate!(%Config{d_model: 256, n_head: 5})
      end
    end

    test "rejects an odd head_dim under RoPE, which would silently drop a channel" do
      assert_raise ArgumentError, ~r/even head_dim/, fn ->
        Config.validate!(%Config{d_model: 18, n_head: 2, pos: :rope})
      end

      # The same shape is fine with learned positions, which do not split heads.
      assert %Config{} = Config.validate!(%Config{d_model: 18, n_head: 2, pos: :learned})
    end

    test "rejects non-positive sizes and out-of-range dropout" do
      assert_raise ArgumentError, ~r/n_layer must be a positive integer/, fn ->
        Config.validate!(%Config{n_layer: 0})
      end

      assert_raise ArgumentError, ~r/dropout must be in/, fn ->
        Config.validate!(%Config{dropout: 1.0})
      end
    end
  end

  test "preset/1 rejects an unknown name" do
    assert_raise ArgumentError, ~r/unknown preset/, fn -> Config.preset("nope") end
  end

  test "swiglu_hidden rounds 8/3 * d_model up to a multiple of 32" do
    assert Config.swiglu_hidden(%Config{d_model: 256}) == 704
    assert rem(Config.swiglu_hidden(%Config{d_model: 384}), 32) == 0
  end
end
