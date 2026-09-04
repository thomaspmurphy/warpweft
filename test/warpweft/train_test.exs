defmodule Warpweft.TrainTest do
  use ExUnit.Case, async: false

  alias Warpweft.{Config, Model, Train}

  doctest Warpweft.Train

  @tiny %Config{
    vocab_size: 64,
    block_size: 16,
    n_layer: 2,
    n_head: 2,
    d_model: 32,
    dropout: 0.0,
    peak_lr: 3.0e-3,
    warmup_steps: 10,
    total_steps: 300,
    clip_norm: 1.0,
    weight_decay: 0.0
  }

  test "loss of random init is close to uniform entropy ln(vocab)" do
    params = Model.init(@tiny, Nx.Random.key(0))
    {x, _} = Nx.Random.randint(Nx.Random.key(1), 0, 64, shape: {4, 16}, type: :s32)
    {y, _} = Nx.Random.randint(Nx.Random.key(2), 0, 64, shape: {4, 16}, type: :s32)

    loss = params |> Model.forward(x, @tiny) |> Train.loss(y) |> Nx.to_number()
    assert_in_delta loss, :math.log(64), 0.5
  end

  @tag :slow
  test "overfits a single batch (end-to-end gradient sanity)" do
    cfg = @tiny
    params = Model.init(cfg, Nx.Random.key(0))
    {opt_init, opt_update} = Train.optimizer(cfg)
    opt_state = opt_init.(params)
    step_fn = Train.build_train_step(cfg, opt_update)

    # one fixed batch with a learnable (repetitive) pattern
    pattern = Stream.cycle(0..7) |> Enum.take(17)
    x = pattern |> Enum.take(16) |> then(&Nx.tensor([&1], type: :s32)) |> Nx.tile([4, 1])
    y = pattern |> Enum.drop(1) |> then(&Nx.tensor([&1], type: :s32)) |> Nx.tile([4, 1])
    key = Nx.Random.key(3)

    {_loss, params, _opt_state} =
      Enum.reduce(1..300, {nil, params, opt_state}, fn _i, {_l, params, opt_state} ->
        step_fn.(params, opt_state, x, y, key)
      end)

    final_loss = params |> Model.forward(x, cfg) |> Train.loss(y) |> Nx.to_number()
    assert final_loss < 0.5
  end

  @tag :slow
  @tag :tmp_dir
  test "checkpoint save/load reproduces identical logits", %{tmp_dir: dir} do
    cfg = @tiny
    params = Model.init(cfg, Nx.Random.key(0))
    Warpweft.Checkpoint.save_best(dir, params, 1.23)

    loaded = Warpweft.Checkpoint.load(Path.join(dir, "best.ckpt"))

    {x, _} = Nx.Random.randint(Nx.Random.key(1), 0, 64, shape: {2, 16}, type: :s32)

    assert Nx.all_close(
             Model.forward(params, x, cfg),
             Model.forward(loaded["params"], x, cfg)
           ) |> Nx.to_number() == 1
  end
end
