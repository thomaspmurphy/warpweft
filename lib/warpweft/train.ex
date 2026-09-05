defmodule Warpweft.Train do
  @moduledoc """
  The training loop, hand-rolled and fully visible:

    1. sample a batch on device
    2. `value_and_grad` of the loss w.r.t. the params map
    3. Polaris update (global-norm clip -> AdamW with warmup+cosine LR)
    4. apply updates

  Steps 2-4 are traced into one jitted function, compiled by XLA exactly
  once (all shapes are fixed), then called `total_steps` times.
  """

  alias Warpweft.{Checkpoint, Config, Model, Schedule}
  alias Warpweft.Data.{Batches, Dataset}

  @doc """
  Mean cross-entropy between logits `{b, t, v}` and target ids `{b, t}`.
  """
  def loss(logits, targets) do
    {b, t, v} = Nx.shape(logits)

    Axon.Losses.categorical_cross_entropy(
      Nx.reshape(targets, {b * t}),
      Nx.reshape(logits, {b * t, v}),
      from_logits: true,
      sparse: true,
      reduction: :mean
    )
  end

  @doc "Gradient-clipped AdamW with warmup+cosine schedule, as a Polaris `{init, update}` pair."
  def optimizer(%Config{} = cfg) do
    schedule =
      Schedule.warmup_cosine(cfg.peak_lr, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio)

    Polaris.Updates.clip_by_global_norm(max_norm: cfg.clip_norm)
    |> Polaris.Updates.compose(
      Polaris.Optimizers.adamw(learning_rate: schedule, decay: cfg.weight_decay)
    )
  end

  @doc "Builds the jitted `(params, opt_state, x, y, key) -> {loss, params, opt_state}` step."
  def build_train_step(%Config{} = cfg, opt_update) do
    Nx.Defn.jit(fn params, opt_state, x, y, key ->
      {loss, grads} =
        Nx.Defn.value_and_grad(params, fn p ->
          p |> Model.forward(x, cfg, key: key) |> loss(y)
        end)

      {updates, opt_state} = opt_update.(grads, opt_state, params)
      {loss, Polaris.Updates.apply_updates(params, updates), opt_state}
    end)
  end

  @doc "Builds the jitted inference-mode `(params, x, y) -> loss` evaluator."
  def build_eval_step(%Config{} = cfg) do
    Nx.Defn.jit(fn params, x, y ->
      params |> Model.forward(x, cfg) |> loss(y)
    end)
  end

  @doc """
  Trains a model from scratch (or resumes with `resume: run_dir`) and
  returns `{params, run_dir}`.
  """
  def run(%Config{} = cfg, opts \\ []) do
    {train_data, val_data, _meta} = Dataset.load(cfg.corpus, cfg.vocab_size)
    tokenizer_dir = Path.join("data/tokenizers", "#{cfg.corpus}-#{cfg.vocab_size}")

    {opt_init, opt_update} = optimizer(cfg)
    train_step = build_train_step(cfg, opt_update)
    eval_step = build_eval_step(cfg)

    {params, opt_state, start_step, run_dir} =
      case Keyword.get(opts, :resume) do
        nil ->
          params = Model.init(cfg, Nx.Random.key(cfg.seed))
          {params, opt_init.(params), 0, Checkpoint.create_run_dir(cfg, tokenizer_dir)}

        run_dir ->
          ckpt = Checkpoint.load(Path.join(run_dir, "latest.ckpt"))
          step = Nx.to_number(ckpt["step"])
          IO.puts("resuming #{run_dir} from step #{step}")
          {ckpt["params"], ckpt["opt_state"], step, run_dir}
      end

    IO.puts(
      "run #{run_dir}: #{Model.param_count(params)} params, " <>
        "#{cfg.total_steps - start_step} steps to go " <>
        "(#{inspect(cfg.pos)}/#{inspect(cfg.norm)}/#{inspect(cfg.mlp)}, tied=#{cfg.tie_embeddings})"
    )

    tokens_per_step = cfg.batch_size * cfg.block_size

    initial = %{
      params: params,
      opt_state: opt_state,
      # Read back rather than reset, so resuming cannot overwrite a better
      # best.ckpt with the first evaluation after the restart.
      best_val: Checkpoint.best_val_loss(run_dir),
      window_start: System.monotonic_time(:millisecond),
      window_steps: 0
    }

    final =
      Batches.stream(train_data, cfg.seed + start_step, cfg.batch_size, cfg.block_size)
      |> Stream.take(cfg.total_steps - start_step)
      |> Stream.with_index(start_step + 1)
      |> Enum.reduce(initial, fn {{x, y, key}, step}, acc ->
        {loss, params, opt_state} = train_step.(acc.params, acc.opt_state, x, y, key)
        acc = %{acc | params: params, opt_state: opt_state, window_steps: acc.window_steps + 1}

        acc =
          if rem(step, cfg.log_every) == 0 or step == cfg.total_steps do
            loss = Nx.to_number(loss)
            now = System.monotonic_time(:millisecond)
            tok_s = acc.window_steps * tokens_per_step * 1000 / max(now - acc.window_start, 1)

            IO.puts(
              "step #{step}/#{cfg.total_steps}  loss #{Float.round(loss, 4)}  " <>
                "#{round(tok_s)} tok/s"
            )

            %{acc | window_start: now, window_steps: 0}
          else
            acc
          end

        if rem(step, cfg.eval_every) == 0 or step == cfg.total_steps do
          val_loss = evaluate(eval_step, params, val_data, cfg)
          IO.puts("step #{step}  val_loss #{Float.round(val_loss, 4)}")
          Checkpoint.save_latest(run_dir, params, opt_state, step)

          if val_loss < acc.best_val do
            Checkpoint.save_best(run_dir, params, val_loss)
            %{acc | best_val: val_loss}
          else
            acc
          end
        else
          acc
        end
      end)

    {final.params, run_dir}
  end

  @doc """
  Converts a loss in nats per token into bits per byte.

  Cross-entropy in nats/token is **not** comparable between models with
  different vocabularies: a tokenizer that packs more text into each token
  earns a higher per-token loss for identical predictive quality. Bits per
  byte divides that out, and is the only fair way to compare a run against
  one that used a different tokenizer.

      iex> Warpweft.Train.bits_per_byte(2.0379, 3.968) |> Float.round(3)
      0.741
  """
  def bits_per_byte(nats_per_token, bytes_per_token) do
    nats_per_token / :math.log(2) / bytes_per_token
  end

  @doc "Mean inference-mode loss over `eval_batches` random validation batches."
  def evaluate(eval_step, params, val_data, %Config{} = cfg) do
    if cfg.eval_batches < 1 do
      raise ArgumentError, "eval_batches must be at least 1, got #{cfg.eval_batches}"
    end

    Batches.stream(val_data, cfg.seed + 7919, cfg.batch_size, cfg.block_size)
    |> Stream.take(cfg.eval_batches)
    |> Enum.map(fn {x, y, _key} -> eval_step.(params, x, y) end)
    |> Enum.map(&Nx.to_number/1)
    |> then(&(Enum.sum(&1) / length(&1)))
  end
end
