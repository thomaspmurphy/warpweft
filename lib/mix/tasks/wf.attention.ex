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
        strict: [run: :string, prompt: :string, layer: :integer, head: :integer, heatmaps: :boolean]
      )

    Mix.Task.run("app.start")

    run_dir = Keyword.get_lazy(opts, :run, &latest_run!/0)
    prompt = Keyword.get(opts, :prompt, "First Citizen:\nBefore we")

    result = Introspect.analyze(run_dir, prompt)
    cfg = result.config

    IO.puts(
      "#{run_dir}: #{cfg.n_layer} layers x #{cfg.n_head} heads, " <>
        "#{length(result.tokens)} tokens, positions=#{cfg.pos}"
    )

    IO.puts("tokens: #{inspect(result.tokens)}")

    layers = if opts[:layer], do: [opts[:layer]], else: 0..(cfg.n_layer - 1) |> Enum.to_list()
    heads = if opts[:head], do: [opts[:head]], else: 0..(cfg.n_head - 1) |> Enum.to_list()
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

  defp latest_run! do
    case "runs" |> File.ls!() |> Enum.sort(:desc) |> List.first() do
      nil -> raise "no runs found; train first with: mix wf.train"
      dir -> Path.join("runs", dir)
    end
  end
end
