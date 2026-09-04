defmodule Warpweft.Introspect do
  @moduledoc """
  Looks inside a trained model's attention.

  Attention weights are a `{batch, head, query, key}` tensor where each
  query row is a probability distribution over the positions it is allowed
  to see. Rather than eyeballing grids, this summarizes each head with a
  few statistics that distinguish the patterns heads reliably specialize
  into:

    * **previous-token** - mass concentrated one step back, the head that
      implements "what came immediately before"
    * **sink** - mass parked on position 0. Heads often dump probability
      on the first token when they have nothing to contribute, because
      softmax forces every row to sum to 1. Reported both raw and as a
      multiple of the uniform baseline: position 0 is visible to all `t`
      query rows while the last position is visible to only one, so even
      a perfectly uniform head puts `H(t)/t` of its mass on position 0.
      Without that correction every diffuse head looks like a sink
    * **self** - mass on the current position, passing the token's own
      representation through
    * **distance** - how far back the head looks on average
    * **entropy** - normalized against a uniform distribution over the
      visible positions, so 0 is a sharp single-position lookup and 1 is
      "attend to everything equally"
  """

  alias Warpweft.{Checkpoint, Config, Model}
  alias Warpweft.Tokenizer.{BPE, Store}

  @shades [{0.6, "█"}, {0.35, "▓"}, {0.15, "▒"}, {0.05, "░"}, {0.005, "·"}]

  @doc """
  Runs `prompt` through the model in `run_dir` and returns
  `%{tokens: [...], stats: [...], weights: [...]}`.
  """
  def analyze(run_dir, prompt) do
    {params, %Config{} = cfg} = Checkpoint.load_run(run_dir)
    bpe = run_dir |> Checkpoint.tokenizer_dir() |> Store.load()

    ids = bpe |> BPE.encode(prompt) |> Enum.take(cfg.block_size)
    tokens = Enum.map(ids, &BPE.decode(bpe, [&1]))

    input = Nx.tensor([ids], type: :s32)
    {_logits, weights} = Model.forward(params, input, cfg, collect_attention: true)

    stats =
      for {layer_weights, layer} <- Enum.with_index(weights),
          head <- 0..(cfg.n_head - 1) do
        layer_weights[[0, head]] |> head_stats() |> Map.merge(%{layer: layer, head: head})
      end

    %{tokens: tokens, stats: stats, weights: weights, config: cfg}
  end

  @doc """
  Statistics for one head's `{t, t}` causal attention matrix.
  """
  def head_stats(w) do
    {t, _} = Nx.shape(w)

    rows = Nx.iota({t, t}, axis: 0)
    cols = Nx.iota({t, t}, axis: 1)
    offset = Nx.subtract(rows, cols)

    mass = fn mask, n -> w |> Nx.multiply(mask) |> Nx.sum() |> Nx.divide(n) |> Nx.to_number() end

    sink = mass.(Nx.equal(cols, 0), t)
    uniform_sink = uniform_sink_mass(t)

    %{
      prev: if(t > 1, do: mass.(Nx.equal(offset, 1), t - 1), else: 0.0),
      sink: sink,
      sink_excess: sink / uniform_sink,
      self: mass.(Nx.equal(offset, 0), t),
      distance: mass.(Nx.select(Nx.greater_equal(offset, 0), offset, 0), t),
      entropy: normalized_entropy(w, t)
    }
  end

  # Mass a uniform causal head puts on position 0: mean over query rows of
  # 1/(row + 1), i.e. the t-th harmonic number over t.
  defp uniform_sink_mass(t) do
    Enum.reduce(1..t, 0.0, fn i, acc -> acc + 1 / i end) / t
  end

  # Mean row entropy divided by the entropy of a uniform distribution over
  # that row's visible positions. Row 0 sees only itself, so it is skipped.
  defp normalized_entropy(_w, t) when t < 2, do: 0.0

  defp normalized_entropy(w, t) do
    row_entropy =
      w
      |> Nx.multiply(Nx.log(Nx.add(w, 1.0e-9)))
      |> Nx.sum(axes: [1])
      |> Nx.negate()

    visible = Nx.iota({t}) |> Nx.add(1) |> Nx.as_type(:f32) |> Nx.log()

    row_entropy
    |> Nx.slice([1], [t - 1])
    |> Nx.divide(Nx.slice(visible, [1], [t - 1]))
    |> Nx.mean()
    |> Nx.to_number()
  end

  @doc """
  One-line label for what a head appears to be doing.

  Order matters: the sink test uses `:sink_excess` (mass relative to the
  uniform baseline) and comes after the sharp-pattern tests, so a merely
  diffuse head is not mislabelled as a sink.
  """
  def classify(%{prev: prev, self: self_, entropy: entropy, distance: distance} = stats) do
    excess = Map.get(stats, :sink_excess, 1.0)

    cond do
      prev > 0.5 -> "previous-token"
      self_ > 0.5 -> "self (current token)"
      excess > 2.0 -> "sink (position 0)"
      entropy > 0.8 -> "diffuse"
      prev > 0.25 -> "mostly previous-token"
      excess > 1.5 -> "partial sink"
      distance > 4.0 -> "long-range (#{Float.round(distance, 1)})"
      true -> "mixed"
    end
  end

  @doc "Prints the per-head statistics table."
  def print_table(%{stats: stats}) do
    IO.puts("\n layer head   prev   sink  vs.unif   self   dist  entropy  pattern")
    IO.puts(" ----- ---- ------ ------ -------- ------ ------ -------- --------------------")

    for s <- stats do
      IO.puts(
        "  #{pad(s.layer, 4)} #{pad(s.head, 4)} " <>
          "#{fmt(s.prev)} #{fmt(s.sink)}    #{num(s.sink_excess)}x " <>
          "#{fmt(s.self)} #{num(s.distance)} " <>
          "#{fmt(s.entropy)}   #{classify(s)}"
      )
    end

    IO.puts("")
  end

  @doc """
  Prints a shaded grid for one head. Rows are query positions (the token
  doing the looking), columns are the positions it attends to.
  """
  def print_heatmap(%{tokens: tokens, weights: weights}, layer, head) do
    w = weights |> Enum.at(layer) |> then(& &1[[0, head]])
    rows = Nx.to_list(w)
    width = tokens |> Enum.map(&String.length(label(&1))) |> Enum.max()

    IO.puts("\nlayer #{layer}, head #{head}  (row = query token, column = attended position)\n")

    for {{row, token}, i} <- Enum.zip(rows, tokens) |> Enum.with_index() do
      cells = row |> Enum.take(i + 1) |> Enum.map_join(&shade/1)
      IO.puts("  #{String.pad_leading(label(token), width)} #{cells}")
    end

    IO.puts("")
  end

  defp shade(weight) do
    Enum.find_value(@shades, " ", fn {threshold, char} -> weight >= threshold && char end)
  end

  defp label(token) do
    token
    |> String.replace("\n", "\\n")
    |> String.replace("\t", "\\t")
    |> String.replace(" ", "_")
  end

  defp fmt(x), do: String.pad_leading(:erlang.float_to_binary(x, decimals: 3), 6)
  defp num(x), do: String.pad_leading(:erlang.float_to_binary(x, decimals: 1), 6)
  defp pad(x, n), do: String.pad_leading(Integer.to_string(x), n)
end
