defmodule Mix.Tasks.Wf.Attention do
  @shortdoc "Inspects what a trained model's attention heads do"
  @moduledoc """
  Runs a prompt through a trained model and reports what each attention
  head attends to.

      mix wf.attention --prompt "First Citizen:"
      mix wf.attention --prompt "ROMEO: What" --heatmaps
      mix wf.attention --run runs/... --layer 0 --head 2

  Without `--layer`/`--head` a summary table for every head is printed.
  `--heatmaps` additionally draws a shaded grid per head; combine with
  `--layer`/`--head` to draw just one.
  """

  use Mix.Task

  alias Warpweft.Introspect

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [
          run: :string,
          prompt: :string,
          layer: :integer,
          head: :integer,
          heatmaps: :boolean
        ]
      )

    Mix.Task.run("app.start")

    run_dir = Warpweft.Runs.resolve!(opts[:run])
    prompt = Keyword.get(opts, :prompt, "First Citizen:\nBefore we")

    result = Introspect.analyze(run_dir, prompt)
    cfg = result.config

    IO.puts(
      "#{run_dir}: #{cfg.n_layer} layers x #{cfg.n_head} heads, " <>
        "#{length(result.tokens)} tokens, positions=#{cfg.pos}"
    )

    IO.puts("tokens: #{inspect(result.tokens)}")

    layers = selection!(opts[:layer], cfg.n_layer, "layer")
    heads = selection!(opts[:head], cfg.n_head, "head")
    selected = for l <- layers, h <- heads, do: {l, h}

    result
    |> Map.update!(:stats, fn stats ->
      Enum.filter(stats, fn s -> {s.layer, s.head} in selected end)
    end)
    |> Introspect.print_table()

    if opts[:heatmaps] do
      for {layer, head} <- selected, do: Introspect.print_heatmap(result, layer, head)
    end

    :ok
  end

  defp selection!(nil, count, _name), do: Enum.to_list(0..(count - 1))

  defp selection!(index, count, _name) when index >= 0 and index < count, do: [index]

  defp selection!(index, count, name) do
    Mix.raise("--#{name} #{index} is out of range: this model has #{count} (0..#{count - 1})")
  end
end
