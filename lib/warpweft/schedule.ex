defmodule Warpweft.Schedule do
  @moduledoc """
  Learning-rate schedule: linear warmup followed by cosine decay to
  `min_lr_ratio * peak`. Polaris has both pieces but no way to compose
  them, so this is written directly.

  Returns a 1-arity function over the (integer tensor) step count, which
  `Polaris.Optimizers.adamw(learning_rate: fun)` accepts as a schedule.
  """

  import Nx.Defn

  def warmup_cosine(peak, warmup_steps, total_steps, min_lr_ratio) do
    &warmup_cosine_impl(&1,
      peak: peak,
      warmup_steps: warmup_steps,
      total_steps: total_steps,
      min_lr_ratio: min_lr_ratio
    )
  end

  defnp warmup_cosine_impl(step, opts \\ []) do
    opts = keyword!(opts, [:peak, :warmup_steps, :total_steps, :min_lr_ratio])
    peak = opts[:peak]
    warmup = opts[:warmup_steps]
    total = opts[:total_steps]
    min_lr = peak * opts[:min_lr_ratio]

    step = Nx.as_type(step, :f32)

    # step + 1 so the very first step is not lr = 0
    warm_lr = peak * Nx.min(step + 1, warmup) / warmup

    progress = Nx.clip((step - warmup) / Nx.max(total - warmup, 1), 0.0, 1.0)
    cosine_lr = min_lr + 0.5 * (peak - min_lr) * (1.0 + Nx.cos(Nx.Constants.pi() * progress))

    Nx.select(step < warmup, warm_lr, cosine_lr)
  end
end
