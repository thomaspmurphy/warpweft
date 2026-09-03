defmodule Warpweft.ScheduleTest do
  use ExUnit.Case, async: true

  alias Warpweft.Schedule

  test "warmup then cosine decay to min ratio" do
    peak = 1.0e-3
    sched = Schedule.warmup_cosine(peak, 100, 1000, 0.1)
    lr = fn step -> sched.(Nx.tensor(step)) |> Nx.to_number() end

    # warmup: rises linearly, never zero
    assert lr.(0) > 0
    assert lr.(50) < lr.(99)
    assert_in_delta lr.(99), peak, peak * 0.02

    # cosine: monotone decreasing after warmup
    assert lr.(200) > lr.(500)
    assert lr.(500) > lr.(900)

    # floor: min ratio at the end, and stays there
    assert_in_delta lr.(1000), peak * 0.1, peak * 0.01
    assert_in_delta lr.(5000), peak * 0.1, peak * 0.01
  end
end
