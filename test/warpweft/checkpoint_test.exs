defmodule Warpweft.CheckpointTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Checkpoint, Config, Model, Train}

  @moduletag :tmp_dir

  @tiny %Config{vocab_size: 32, block_size: 8, n_layer: 1, n_head: 2, d_model: 16, dropout: 0.0}

  defp params, do: Model.init(@tiny, Nx.Random.key(0))

  test "save_latest round-trips params, optimiser state and step", %{tmp_dir: dir} do
    params = params()
    {init, update} = Train.optimizer(@tiny)
    opt_state = init.(params)

    # Take a step so the optimiser state holds non-trivial moments.
    grad = Map.new(params, fn {k, v} -> {k, scale_leaves(v)} end)
    {_updates, opt_state} = update.(grad, opt_state, params)

    Checkpoint.save_latest(dir, params, opt_state, 1234)
    loaded = Checkpoint.load(Path.join(dir, "latest.ckpt"))

    assert Nx.to_number(loaded["step"]) == 1234

    {x, _} = Nx.Random.randint(Nx.Random.key(1), 0, 32, shape: {1, 8}, type: :s32)

    assert Nx.all_close(
             Model.forward(params, x, @tiny),
             Model.forward(loaded["params"], x, @tiny)
           )
           |> Nx.to_number() == 1

    # The optimiser state must survive too, or resuming silently restarts
    # Adam's moment estimates and the step count it bases bias correction on.
    {a, _} = update.(grad, opt_state, params)
    {b, _} = update.(grad, loaded["opt_state"], params)
    assert Nx.all_close(a["wte"]["kernel"], b["wte"]["kernel"], atol: 1.0e-6) |> Nx.to_number() == 1
  end

  defp scale_leaves(%Nx.Tensor{} = t), do: Nx.multiply(Nx.broadcast(0.01, Nx.shape(t)), 1.0)
  defp scale_leaves(map), do: Map.new(map, fn {k, v} -> {k, scale_leaves(v)} end)

  describe "best_val_loss/1" do
    test "reads back what save_best wrote", %{tmp_dir: dir} do
      Checkpoint.save_best(dir, params(), 1.75)
      assert_in_delta Checkpoint.best_val_loss(dir), 1.75, 1.0e-5
    end

    # Returning :infinity here is what used to make the first evaluation
    # after a resume overwrite a better checkpoint.
    test "is :infinity when no best checkpoint exists", %{tmp_dir: dir} do
      assert Checkpoint.best_val_loss(dir) == :infinity
    end
  end

  describe "load_run/1" do
    test "prefers best.ckpt over latest.ckpt", %{tmp_dir: dir} do
      Config.save(@tiny, dir)
      best = params()
      other = Model.init(@tiny, Nx.Random.key(99))

      Checkpoint.save_best(dir, best, 1.0)
      Checkpoint.save_latest(dir, other, %{}, 10)

      {loaded, config} = Checkpoint.load_run(dir)
      assert config == @tiny

      assert Nx.all_close(loaded["wte"]["kernel"], best["wte"]["kernel"]) |> Nx.to_number() == 1
    end

    test "falls back to latest.ckpt", %{tmp_dir: dir} do
      Config.save(@tiny, dir)
      Checkpoint.save_latest(dir, params(), %{}, 5)

      {loaded, _config} = Checkpoint.load_run(dir)
      assert is_map(loaded)
    end

    test "raises when there is no checkpoint at all", %{tmp_dir: dir} do
      Config.save(@tiny, dir)
      assert_raise RuntimeError, ~r/no checkpoint found/, fn -> Checkpoint.load_run(dir) end
    end
  end
end
