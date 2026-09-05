defmodule Warpweft.OptimizerTest do
  use ExUnit.Case, async: true

  alias Warpweft.{Config, Train}

  # Nothing else connects Warpweft.Schedule to Polaris: the schedule is
  # tested in isolation, and the training tests only check that loss goes
  # down, which it does at any sane fixed learning rate. Replacing the
  # schedule with a constant, or dropping weight decay, previously left
  # the whole suite green.
  #
  # Polaris also silently ignores unrecognised optimiser options, so a
  # renamed or mistyped key would vanish without error. These tests drive
  # the real optimiser and watch the updates it produces.

  defp cfg(overrides \\ []) do
    struct!(
      %Config{peak_lr: 1.0, warmup_steps: 10, total_steps: 100, min_lr_ratio: 0.1, weight_decay: 0.0},
      overrides
    )
  end

  # Adam normalises by the gradient's own running magnitude, so with a
  # constant gradient the update size converges to the learning rate.
  defp update_sizes(config, steps) do
    {init, update} = Train.optimizer(config)
    params = %{"w" => Nx.tensor([1.0, 1.0, 1.0])}
    grad = %{"w" => Nx.tensor([1.0, 1.0, 1.0])}

    {sizes, _state} =
      Enum.map_reduce(1..steps, init.(params), fn _i, state ->
        {updates, state} = update.(grad, state, params)
        size = updates["w"] |> Nx.abs() |> Nx.mean() |> Nx.to_number()
        {size, state}
      end)

    sizes
  end

  test "the warmup ramp is actually wired into the optimizer" do
    [first, second, third | _] = update_sizes(cfg(), 3)

    # Warmup is linear over 10 steps at peak 1.0, so ~0.1, 0.2, 0.3.
    assert_in_delta first, 0.1, 0.02
    assert_in_delta second, 0.2, 0.02
    assert_in_delta third, 0.3, 0.02

    assert second > first and third > second,
           "learning rate is not ramping: the schedule is not reaching Polaris"
  end

  test "the cosine decay is wired in too" do
    sizes = update_sizes(cfg(), 60)

    peak = Enum.at(sizes, 9)
    later = Enum.at(sizes, 39)
    latest = Enum.at(sizes, 59)

    assert_in_delta peak, 1.0, 0.05
    assert later < peak * 0.95, "learning rate did not decay after warmup"
    assert latest < later, "decay is not monotonic after warmup"
  end

  test "weight decay reaches the optimizer" do
    without = update_sizes(cfg(weight_decay: 0.0), 5)
    with_decay = update_sizes(cfg(weight_decay: 0.5), 5)

    refute without == with_decay,
           "identical updates with and without weight decay: :decay is not being applied"
  end

  test "gradient clipping bounds the update when gradients explode" do
    config = cfg(clip_norm: 1.0)
    {init, update} = Train.optimizer(config)

    params = %{"w" => Nx.tensor([1.0, 1.0, 1.0])}
    huge = %{"w" => Nx.tensor([1.0e6, 1.0e6, 1.0e6])}
    tiny = %{"w" => Nx.tensor([1.0, 1.0, 1.0])}

    {huge_update, _} = update.(huge, init.(params), params)
    {tiny_update, _} = update.(tiny, init.(params), params)

    # After clipping to a fixed global norm, Adam sees the same direction
    # and magnitude regardless of how large the raw gradient was.
    assert Nx.all_close(huge_update["w"], tiny_update["w"], atol: 1.0e-4) |> Nx.to_number() == 1,
           "a 10^6 gradient produced a different update from a unit one: clipping is not applied"
  end
end
